"""Train-only density selection, retaining coarse HMM-occupancy scenarios as sensitivity."""
import hashlib
import json

import hdbscan
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .advanced import hdbscan_walk
from .data import ROOT, config, save_json
from .extensions import summarize
from .external import DEST


def run_density():
    cfg = config()
    protocol = json.loads((DEST / "protocol.json").read_text(encoding="utf-8"))
    name = "regime/hdbscan_density.py"
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256((protocol["fingerprint"]+json.dumps(hashes, sort_keys=True)).encode()).hexdigest()
    completed = [(row.asset, row.model) for row in pd.read_csv(DEST / "model_scores.csv").itertuples()
                 if row.model != "HDBSCAN_DENSITY"]
    for asset in cfg["assets"]:
        target = DEST / f"{asset}_HDBSCAN_DENSITY.parquet"
        receipt = target.with_suffix(".json")
        cached = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        if cached.get("fingerprint") == fingerprint and target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == cached["sha256"]:
            completed.append((asset, "HDBSCAN_DENSITY"))
            continue
        frame = pd.read_parquet(DEST / f"{asset}_features.parquet")
        columns = cfg["core_features"]
        train = frame.loc[:cfg["train_end"], columns].dropna()
        x = StandardScaler().fit_transform(train)
        memory = Memory(str(ROOT / f".tools/hdbscan-{asset}"), verbose=0)
        candidates = []
        # Exhaust all possible minimum sizes allowing at least two clusters:
        # lower=2 (a cluster contains >=2 points), upper=floor(n/2), no arbitrary grid spacing.
        with threadpool_limits(limits=1):
            for size in range(2, len(train)//2+1):
                fitted = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=len(columns)+1,
                    gen_min_span_tree=True, memory=memory).fit(x)
                clusters = len(set(fitted.labels_)-{-1})
                candidates.append({"min_cluster_size": size, "clusters": clusters,
                    "train_relative_validity": fitted.relative_validity_ if clusters >= 2 else np.nan,
                    "train_noise_fraction": np.mean(fitted.labels_ == -1)})
        table = pd.DataFrame(candidates)
        table.to_csv(DEST / f"{asset}_HDBSCAN_density_selection.csv", index=False)
        valid = table.dropna(subset=["train_relative_validity"])
        if valid.empty:
            raise ValueError("No multi-cluster HDBSCAN candidate: report this rather than invent a regime")
        best = valid.sort_values(["train_relative_validity", "min_cluster_size"]).iloc[-1]
        size = int(best.min_cluster_size)
        validation, _ = hdbscan_walk(frame.loc[:cfg["validation_end"]], columns,
            cfg["validation_start"], cfg["validation_end"], size)
        prediction, diagnostics = hdbscan_walk(frame, columns, cfg["test_start"], cfg["test_end"], size)
        validation.to_parquet(DEST / f"{asset}_HDBSCAN_DENSITY_validation.parquet")
        prediction.to_parquet(target)
        save_json(receipt, {"fingerprint": fingerprint, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "decision": {"min_cluster_size": size, "min_samples": len(columns)+1,
                         "train_relative_validity": best.train_relative_validity,
                         "validation_noise_fraction": validation.noise.mean(),
                         "selection": "Max train relative_validity_ over all feasible minimum sizes; exact ties prefer larger size. No test tuning."},
            "diagnostics": diagnostics})
        completed.append((asset, "HDBSCAN_DENSITY"))
        print(f"Density selection {asset}: min_cluster_size={size}, score={best.train_relative_validity:.6g}", flush=True)
    protocol["density_addendum"] = {"fingerprint": fingerprint, "hashes": hashes,
        "criterion": "Train-only relative_validity_ (DBCV approximation), not ground-truth accuracy"}
    save_json(DEST / "protocol.json", protocol)
    summarize(sorted(completed))
    run = json.loads((DEST / "run.json").read_text(encoding="utf-8"))
    run.update(models=len(completed), density_fingerprint=fingerprint)
    save_json(DEST / "run.json", run)


if __name__ == "__main__":
    from .reproduce import seal_extensions
    run_density()
    seal_extensions()
