"""Nominal USD-pegged circulating supply, kept distinct from market capitalization."""
import hashlib
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import ROOT, config, save_json
from .external import DEST, asof_panel, canonical_hash, feature_observations, validate_observations

COVARIATES = ["stablecoin_supply_growth", "defi_tvl_growth", "market_cap_growth"]


def add_supply(panel):
    with zipfile.ZipFile(ROOT / "data/snapshots/external_raw.zip") as archive:
        rows = json.loads(archive.read("stablecoins"))
    observations = pd.DataFrame({"observation_date": pd.to_datetime([int(row["date"]) for row in rows], unit="s", utc=True),
        "value": [row.get("totalCirculating", {}).get("peggedUSD") for row in rows]})
    observations = observations.loc[observations.observation_date.between(
        pd.Timestamp(config()["raw_start"], tz="UTC"), pd.Timestamp(config()["end"], tz="UTC"))].dropna()
    observations["available_date"] = observations.observation_date + pd.Timedelta(days=1)
    observations["series"], observations["unit"], observations["source"] = "stablecoin_supply", "nominal USD at peg", "DefiLlama/totalCirculating.peggedUSD"
    observations = validate_observations(observations)
    expected = ROOT / "data/snapshots/supply_source.json"
    digest = canonical_hash(observations)
    if expected.exists() and json.loads(expected.read_text(encoding="utf-8"))["sha256"] != digest:
        raise ValueError("Historical stablecoin supply changed")
    if not expected.exists():
        save_json(expected, {"sha256": digest, "observations": len(observations), "source": "https://stablecoins.llama.fi/stablecoincharts/all",
            "field": "totalCirculating.peggedUSD", "unit": "USD face value at peg; not mark-to-market USD capitalization"})
    addition = asof_panel(pd.concat([observations, feature_observations(observations)]), panel.index)
    panel = panel.drop(columns=list(addition.columns), errors="ignore").join(addition)
    panel.to_parquet(DEST / "external_panel.parquet")
    return panel


def _supply_worker(job):
    # Reuse the same matched-cohort estimator and artifact contract, with one additional covariate group.
    from .extensions import GROUPS, _worker
    GROUPS["SUPPLY"] = COVARIATES
    asset, label = _worker(job)
    # statsmodels may emit a probability one ULP above one. Normalize conditional
    # weights before downstream strict long/cash accounting; density is unchanged.
    maximum = 0.
    for suffix in ("", "_validation"):
        path = DEST / f"{asset}_{label}{suffix}.parquet"
        prediction = pd.read_parquet(path)
        columns = [column for column in prediction if column.startswith("p_") and column[2:].isdigit()]
        weights = prediction[columns]
        if not np.isfinite(weights).all().all() or (weights < 0).any().any() or (weights.sum(axis=1) <= 0).any():
            raise ValueError("Invalid filtered conditional weights")
        maximum = max(maximum, float((weights.sum(axis=1)-1).abs().max()))
        prediction[columns] = weights.div(weights.sum(axis=1), axis=0)
        prediction["p_high_vol"] = prediction[columns[-1]]
        prediction.to_parquet(path)
    receipt = DEST / f"{asset}_{label}.json"
    record = json.loads(receipt.read_text(encoding="utf-8"))
    record.update(normalization_maximum_mass_error=maximum,
                  sha256=hashlib.sha256((DEST / f"{asset}_{label}.parquet").read_bytes()).hexdigest())
    save_json(receipt, record)
    return asset, label


def run_supply():
    from .extensions import summarize
    panel = add_supply(pd.read_parquet(DEST / "external_panel.parquet"))
    protocol_path = DEST / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("regime/supply.py", "data/snapshots/supply_source.json")}
    fingerprint = hashlib.sha256((protocol["fingerprint"] + json.dumps(hashes, sort_keys=True)).encode()).hexdigest()
    audit = []
    for asset in config()["assets"]:
        path = DEST / f"{asset}_features.parquet"
        frame = pd.read_parquet(path)
        frame["stablecoin_supply_growth_lag2"] = panel.stablecoin_supply_growth.shift(config()["execution_delay_days"])
        frame.to_parquet(path)
        clean = frame.dropna(subset=[column+"_lag2" for column in COVARIATES])
        for split, start, end in [("train", config()["analysis_start"], config()["train_end"]),
                ("validation", config()["validation_start"], config()["validation_end"]),
                ("test", config()["test_start"], config()["test_end"])]:
            period = clean.loc[start:end]
            audit.append({"asset": asset, "group": "SUPPLY", "split": split, "days": len(period),
                          "first_date": period.index.min(), "last_date": period.index.max()})
    old_audit = pd.read_csv(DEST / "sample_counts.csv")
    pd.concat([old_audit[old_audit.group.ne("SUPPLY")], pd.DataFrame(audit)]).to_csv(DEST / "sample_counts.csv", index=False)
    protocol["supply_addendum"] = {"hashes": hashes, "fingerprint": fingerprint, "covariates": COVARIATES,
        "reason": "Separate pure nominal supply growth from mark-to-market stablecoin capitalization; no price duplication"}
    save_json(protocol_path, protocol)
    prior = pd.read_csv(DEST / "model_scores.csv")
    completed = [(row.asset, row.model) for row in prior.itertuples() if not row.model.endswith("_SUPPLY")]
    jobs = [(asset, kind, None, fingerprint) for asset in config()["assets"] for kind in ("MS0_SUPPLY", "TVTP_SUPPLY")]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for asset, label in pool.map(_supply_worker, jobs):
            completed.append((asset, label))
            print(f"Completed supply: {asset} {label}", flush=True)
    summarize(sorted(completed))
    run = json.loads((DEST / "run.json").read_text(encoding="utf-8"))
    run.update(models=len(completed), supply_fingerprint=fingerprint)
    save_json(DEST / "run.json", run)
