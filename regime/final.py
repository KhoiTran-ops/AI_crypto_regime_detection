"""Locked holdout evaluation and compact artifacts shared by paper and dashboard."""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .benchmarks import cluster_walk, markov_walk
from .data import ROOT, config, save_json
from .evaluation import portfolio_comparison
from .models import fit_hmm, walk_forward
from .research import dataset, input_fingerprint, load_selection, state_diagnostics

FINAL = ROOT / "results"


def mean_difference_ci(left, right, repetitions=1000, block=30):
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    delta = (joined.left - joined.right).to_numpy()
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(delta), int(np.ceil(len(delta) / block)))
        positions = ((starts[:, None] + np.arange(block)) % len(delta)).ravel()[:len(delta)]
        draws.append(delta[positions].mean())
    return {"n_common": len(delta), "mean_difference": delta.mean(),
            "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975),
            "block_days": block, "bootstrap_repetitions": repetitions}


def freeze():
    selected = load_selection()
    if not all(choice.get("adequacy", {}).get("recovery_passed") for choice in selected["choices"].values()):
        raise ValueError("Core sample recovery has not passed. Final test stays closed.")
    research = ROOT / "outputs/research"
    benchmarks = json.loads((research / "benchmark_selection.json").read_text())
    cohorts = json.loads((research / "cohort_selection.json").read_text())
    for item in (benchmarks, cohorts):
        if item["input_fingerprint"] != input_fingerprint():
            raise ValueError("Selection inputs differ from current snapshot")
    scientific_code = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for name in ("data", "derivatives", "features", "models", "benchmarks", "evaluation", "research", "adequacy", "final")
                       if (path := ROOT / f"regime/{name}.py").exists()}
    protocol = {"core": selected["choices"], "benchmarks": benchmarks["choices"],
                "cohorts": cohorts["choices"], "input_fingerprint": input_fingerprint(),
                "scientific_code": scientific_code, "config": config(),
                "requirements_hash": hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest(),
                "test_policy": "Monthly expanding refits. No test-based selection. All periods/costs reported.",
                "execution": "Signal at UTC close t; trade open t+2. Final liquidation at final test open.",
                "costs": "5/10/25 bps per traded notional; 10 bps primary; cash rate 0, no short/leverage.",
                "targets": "Joint log density only within identical feature sets; return density separately.",
                "uncertainty": "Paired circular block bootstrap, 30-day blocks, 1000 repetitions; sensitivity 7/60.",
                "limitations": ["Retrospective archives, not point-in-time vintages",
                    "Gaussian diagonal emissions; overlapping rolling-volatility features",
                    "Exploratory B/C extensions have no parameter-recovery certification"]}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    FINAL.mkdir(exist_ok=True)
    path = FINAL / "protocol_lock.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["protocol_hash"] != fingerprint:
            raise ValueError("Locked protocol changed; preserve this run and use an explicitly new version")
        return prior
    record = {"protocol_hash": fingerprint, "locked_before_test_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol}
    save_json(path, record)
    return record


def evaluate():
    lock, cfg = freeze(), config()
    protocol = lock["protocol"]
    start, end = cfg["test_start"], cfg["test_end"]
    scores, performance_rows, comparisons, uncertainty, stability = [], [], [], [], []
    all_predictions = {}
    for asset in cfg["assets"]:
        frame, columns = dataset(asset)
        selected_k = protocol["core"][asset]["k"]
        asset_predictions = {}
        specs = [("HMM", "A", selected_k), ("GMM", "A", protocol["benchmarks"][asset]["GMM"]),
                 ("KMeans", "A", selected_k), ("HMM_return", "A", protocol["benchmarks"][asset]["HMM_return"]),
                 ("MSAR", "A", protocol["benchmarks"][asset]["MSAR"]),
                 ("HMM_B", "B", protocol["cohorts"][asset]["B"]),
                 ("HMM_C", "C", protocol["cohorts"][asset]["C"]),
                 ("MS0", "B", 2), ("NHHMM", "B", 2)]
        for kind, cohort, k in specs:
            print(f"Holdout {asset} {kind} K={k}", flush=True)
            source, features = dataset(asset, cohort)
            path = FINAL / f"{asset}_{kind}.parquet"
            if path.exists():
                prediction = pd.read_parquet(path)
                diagnostics = json.loads(path.with_suffix(".diagnostics.json").read_text())
            else:
                coefficients = None
                if kind.startswith("HMM"):
                    features = ["return"] if kind == "HMM_return" else features
                    prediction, diagnostics, _ = walk_forward(source, features, k, start, end,
                        cfg["refit_seeds"], initial_seeds=cfg["seeds"])
                elif kind in ("GMM", "KMeans"):
                    prediction, diagnostics = cluster_walk(source, features, kind, k, start, end)
                else:
                    covariates = None
                    if kind in ("MS0", "NHHMM"):
                        source["funding_lag2"] = source.funding.shift(2)
                        source["volume_lag2"] = source.volume_z.shift(2)
                        source = source.dropna(subset=["return", "funding_lag2", "volume_lag2"])
                        if kind == "NHHMM":
                            covariates = ["funding_lag2", "volume_lag2"]
                    prediction, diagnostics, coefficients = markov_walk(source, kind, k, start, end, covariates)
                prediction.to_parquet(path)
                save_json(path.with_suffix(".diagnostics.json"), diagnostics)
                if coefficients is not None:
                    coefficients.to_csv(path.with_suffix(".coefficients.csv"), index=False)
            if not (pd.to_datetime(prediction.fit_cutoff, utc=True) < prediction.index).all():
                raise ValueError("A model used data at or after its prediction date")
            probability_columns = [column for column in prediction if column.startswith("p_") and column != "p_high_vol"]
            if not np.allclose(prediction[probability_columns].sum(axis=1), 1, atol=1e-6):
                raise ValueError("Regime probabilities do not sum to one")
            asset_predictions[kind] = prediction
            group = "joint_core" if kind in ("HMM", "GMM") else (
                "return_only_A" if kind in ("MSAR", "HMM_return") else
                "return_only_B" if kind in ("MS0", "NHHMM") else f"joint_{cohort}")
            nonconverged = sum(not item.get("converged", item.get("converged_starts", 0) > 0) for item in diagnostics)
            scores.append({"asset": asset, "model": kind, "cohort": cohort, "k": k,
                "score_group": group, "test_days": len(prediction),
                "mean_log_score": prediction.log_score.mean(),
                "mean_return_log_score": (prediction["return_log_score"].mean() if "return_log_score" in prediction
                    else prediction.log_score.mean() if kind in ("MSAR", "MS0", "NHHMM") else np.nan),
                "nonconverged_refits": nonconverged,
                "state_changes": int(prediction.state.diff().fillna(0).ne(0).sum()),
                "mean_high_vol_probability": prediction.p_high_vol.mean()})
            metrics, paths, ci = portfolio_comparison(frame, prediction, start, end,
                costs=cfg["cost_bps"], target=cfg["volatility_target"])
            metrics.insert(0, "asset", asset)
            metrics.insert(1, "model", kind)
            performance_rows.extend(metrics.to_dict("records"))
            for name, result in paths.items():
                result.to_parquet(FINAL / f"{asset}_{kind}_{name}_portfolio.parquet")
            uncertainty.append({"asset": asset, "model": kind, "comparison": "regime_aware minus vol_target", **ci})
            if kind == "HMM":
                for block in (7, 60):
                    from .evaluation import paired_bootstrap
                    uncertainty.append({"asset": asset, "model": kind, "comparison": "regime_aware minus vol_target",
                        **paired_bootstrap(paths["regime_aware"].net_return,
                                           paths["vol_target"].net_return, block=block)})
        for left, right in (("HMM", "GMM"), ("HMM_return", "MSAR"), ("NHHMM", "MS0")):
            comparisons.append({"asset": asset, "left": left, "right": right,
                **mean_difference_ci(asset_predictions[left].log_score, asset_predictions[right].log_score)})
        for extension in ("HMM_B", "HMM_C"):
            comparisons.append({"asset": asset, "left": extension, "right": "HMM",
                "target": "marginal_return_predictive_density",
                **mean_difference_ci(asset_predictions[extension].return_log_score,
                                     asset_predictions["HMM"].return_log_score)})
        for other in ("GMM", "KMeans"):
            common = pd.concat([asset_predictions["HMM"].state.rename("hmm"),
                                asset_predictions[other].state.rename("other")], axis=1).dropna()
            stability.append({"asset": asset, "benchmark": other, "n_common": len(common),
                              "adjusted_rand_index": adjusted_rand_score(common.hmm, common.other)})
        train = frame.loc[:cfg["validation_end"]]
        initial = fit_hmm(train, columns, selected_k, cfg["seeds"])
        save_json(FINAL / f"{asset}_initial_states.json",
            {"states": state_diagnostics(initial, train), "metadata": initial.metadata,
             "transition": initial.transition.tolist(), "stress_state": initial.stress_state,
             "meaning": "Model fit through 2024-12-31; profiles are descriptive, not known true labels"})
        frame.to_parquet(FINAL / f"{asset}_features.parquet")
        all_predictions[asset] = asset_predictions
        pd.DataFrame(scores).to_csv(FINAL / "model_scores.csv", index=False)
        pd.DataFrame(performance_rows).to_csv(FINAL / "portfolio_performance.csv", index=False)
        pd.DataFrame(comparisons).to_csv(FINAL / "score_comparisons.csv", index=False)
        pd.DataFrame(uncertainty).to_csv(FINAL / "portfolio_uncertainty.csv", index=False)
        pd.DataFrame(stability).to_csv(FINAL / "method_comparison.csv", index=False)
    save_json(FINAL / "run.json", {"status": "completed", "protocol_hash": lock["protocol_hash"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": end, "test_start": start, "assets": cfg["assets"],
        "execution": protocol["execution"], "data_fingerprint": input_fingerprint()})
    return all_predictions
