"""Data cohorts, pre-test model selection and immutable result artifacts."""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import ROOT, config, save_json
from .features import CORE_FEATURES, build_features
from .models import fit_hmm, walk_forward


def dataset(asset, cohort="A"):
    frame = build_features(pd.read_parquet(ROOT / f"data/processed/{asset}_spot.parquet").loc["2017-08-18":])
    frame = frame.loc[config()["analysis_start"]:]
    columns = CORE_FEATURES.copy()
    if cohort in ("B", "C"):
        frame = frame.join(pd.read_parquet(ROOT / f"data/processed/{asset}_funding.parquet"))
        columns += ["funding"]
        frame = frame.loc["2019-11-28":]
    if cohort == "C":
        oi = pd.read_parquet(ROOT / f"data/processed/{asset}_oi.parquet")
        oi["oi_change"] = np.log(oi.oi / oi.oi.shift(1))
        frame = frame.join(oi)
        frame = frame.loc["2021-12-02":]
        columns += ["oi_change"]
    return frame, columns


def input_fingerprint():
    paths = [ROOT / "config.json", *sorted((ROOT / "data/processed").glob("*.parquet"))]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def state_diagnostics(fitted, train):
    clean = train.dropna(subset=fitted.columns)
    probabilities, _ = fitted.filter(clean)
    rows = []
    for state in range(len(fitted.order)):
        weights = probabilities[f"p_{state}"]
        mean = float(np.average(clean["return"], weights=weights))
        variance = float(np.average((clean["return"] - mean) ** 2, weights=weights))
        assigned = probabilities.state.eq(state)
        episodes = int((assigned & ~assigned.shift(1, fill_value=False)).sum())
        persistence = float(fitted.transition[state, state])
        rows.append({"state": state, "posterior_days": float(weights.sum()), "occupancy": float(weights.mean()),
                     "argmax_episodes": episodes, "mean_return_daily": mean,
                     "volatility_annual": np.sqrt(variance * 365), "persistence": persistence,
                     "expected_duration_days": 1 / (1 - persistence) if persistence < 1 else None,
                     "limited_evidence": bool(weights.sum() < 60 or episodes < 3)})
    return rows


def select():
    cfg = config()
    folder = ROOT / "outputs/research"
    folder.mkdir(parents=True, exist_ok=True)
    choices, summaries = {}, []
    for asset in cfg["assets"]:
        frame, columns = dataset(asset)
        choices[asset] = {}
        for k in cfg["k"]:
            print(f"Validation {asset}, K={k}", flush=True)
            prediction, fits, last_model = walk_forward(
                frame.loc[:cfg["validation_end"]], columns, k, cfg["validation_start"],
                cfg["validation_end"], cfg["refit_seeds"], cfg["max_iter"],
                initial_seeds=cfg["seeds"])
            prediction.to_parquet(folder / f"{asset}_validation_k{k}.parquet")
            initial = fit_hmm(frame.loc[:cfg["train_end"]], columns, k, cfg["seeds"], cfg["max_iter"])
            diagnostics = state_diagnostics(initial, frame.loc[:cfg["train_end"]])
            row = {"asset": asset, "model": "HMM", "k": k,
                   "validation_log_score": float(prediction.log_score.mean()),
                   "validation_days": len(prediction),
                   "train_bic": initial.metadata["bic"], "train_aic": initial.metadata["aic"],
                   "n_parameters": initial.metadata["n_parameters"],
                   "converged_starts": initial.metadata["converged_starts"],
                   "min_posterior_days": min(item["posterior_days"] for item in diagnostics),
                   "limited_states": sum(item["limited_evidence"] for item in diagnostics)}
            summaries.append(row)
            save_json(folder / f"{asset}_k{k}_diagnostics.json",
                      {"train": initial.metadata, "states": diagnostics, "validation_refits": fits})
            print(json.dumps(row), flush=True)
            pd.DataFrame(summaries).to_csv(folder / "model_selection.csv", index=False)
        scores = [row for row in summaries if row["asset"] == asset]
        winner = max(scores, key=lambda row: row["validation_log_score"])
        choices[asset] = {"k": winner["k"], "selection_criterion": "mean monthly walk-forward validation log density",
                          "limited_states": winner["limited_states"]}
    save_json(folder / "selection.json", {"choices": choices, "config": cfg,
              "input_fingerprint": input_fingerprint(),
              "selected_at_utc": datetime.now(timezone.utc).isoformat(),
              "final_test_opened": False})
    return choices


def load_selection():
    selected = json.loads((ROOT / "outputs/research/selection.json").read_text())
    if selected["input_fingerprint"] != input_fingerprint():
        raise ValueError("Data/config changed after selection. Rerun selection before final evaluation.")
    return selected


def benchmark_selection():
    from .benchmarks import cluster_walk, markov_walk
    cfg = config()
    folder = ROOT / "outputs/research"
    rows, decisions = [], {}
    for asset in cfg["assets"]:
        frame, columns = dataset(asset)
        decisions[asset] = {}
        for k in cfg["k"]:
            path = folder / f"{asset}_GMM_validation_k{k}.parquet"
            if path.exists():
                prediction = pd.read_parquet(path)
            else:
                prediction, diagnostics = cluster_walk(frame, columns, "GMM", k,
                                                       cfg["validation_start"], cfg["validation_end"])
                prediction.to_parquet(path)
                save_json(path.with_suffix(".json"), diagnostics)
            rows.append({"asset": asset, "model": "GMM", "k": k,
                         "target": "joint_core_features", "validation_log_score": prediction.log_score.mean()})
        best = max([item for item in rows if item["asset"] == asset and item["model"] == "GMM"],
                   key=lambda item: item["validation_log_score"])
        decisions[asset]["GMM"] = int(best["k"])
        for kind in ("HMM_return", "MSAR"):
            for k in (2, 3):
                path = folder / f"{asset}_{kind}_validation_k{k}.parquet"
                if path.exists():
                    prediction = pd.read_parquet(path)
                elif kind == "HMM_return":
                    prediction, diagnostics, _ = walk_forward(frame, ["return"], k, cfg["validation_start"],
                        cfg["validation_end"], cfg["refit_seeds"], initial_seeds=cfg["seeds"])
                    prediction.to_parquet(path)
                    save_json(path.with_suffix(".json"), diagnostics)
                else:
                    print(f"Validation {asset} {kind} K={k}", flush=True)
                    prediction, diagnostics, coefficients = markov_walk(frame, kind, k,
                        cfg["validation_start"], cfg["validation_end"])
                    prediction.to_parquet(path)
                    save_json(path.with_suffix(".json"), diagnostics)
                    coefficients.to_csv(path.with_suffix(".coefficients.csv"), index=False)
                rows.append({"asset": asset, "model": kind, "k": k,
                             "target": "return_only", "validation_log_score": prediction.log_score.mean()})
                pd.DataFrame(rows).to_csv(folder / "benchmark_selection.csv", index=False)
            candidates = [item for item in rows if item["asset"] == asset and item["model"] == kind]
            decisions[asset][kind] = int(max(candidates, key=lambda item: item["validation_log_score"])["k"])
        funding_frame, _ = dataset(asset, "B")
        funding_frame["funding_lag2"] = funding_frame.funding.shift(2)
        funding_frame["volume_lag2"] = funding_frame.volume_z.shift(2)
        for kind in ("MS0", "NHHMM"):
            covariates = ["funding_lag2", "volume_lag2"] if kind == "NHHMM" else None
            # Both models must use precisely the same rows, including lag warm-up.
            common = funding_frame.dropna(subset=["return", "funding_lag2", "volume_lag2"])
            path = folder / f"{asset}_{kind}_validation.parquet"
            if path.exists():
                prediction = pd.read_parquet(path)
            else:
                print(f"Validation {asset} {kind} on funding cohort", flush=True)
                prediction, diagnostics, coefficients = markov_walk(common, kind, 2,
                    cfg["validation_start"], cfg["validation_end"], covariates)
                prediction.to_parquet(path)
                save_json(path.with_suffix(".json"), diagnostics)
                coefficients.to_csv(path.with_suffix(".coefficients.csv"), index=False)
            rows.append({"asset": asset, "model": kind, "k": 2, "target": "return_only_funding_cohort",
                         "validation_log_score": prediction.log_score.mean()})
            pd.DataFrame(rows).to_csv(folder / "benchmark_selection.csv", index=False)
    save_json(folder / "benchmark_selection.json", {"choices": decisions, "final_test_opened": False,
                                                   "input_fingerprint": input_fingerprint()})
    return decisions



def cohort_selection():
    """Funding/OI extensions are exploratory; their K is selected before opening test."""
    cfg, choices, rows = config(), {}, []
    folder = ROOT / "outputs/research"
    for asset in cfg["assets"]:
        choices[asset] = {}
        for cohort in ("B", "C"):
            frame, columns = dataset(asset, cohort)
            start = "2023-01-01" if cohort == "B" else "2024-01-01"
            for k in (2, 3):
                print(f"Validation {asset} cohort {cohort} K={k}", flush=True)
                prediction, diagnostics, _ = walk_forward(
                    frame.loc[:cfg["validation_end"]], columns, k, start,
                    cfg["validation_end"], cfg["refit_seeds"], initial_seeds=cfg["seeds"])
                prediction.to_parquet(folder / f"{asset}_{cohort}_validation_k{k}.parquet")
                save_json(folder / f"{asset}_{cohort}_validation_k{k}.json", diagnostics)
                initial = fit_hmm(frame.loc[frame.index < pd.Timestamp(start, tz="UTC")], columns,
                                  k, cfg["seeds"])
                states = state_diagnostics(initial, frame.loc[frame.index < pd.Timestamp(start, tz="UTC")])
                rows.append({"asset": asset, "cohort": cohort, "k": k,
                    "validation_start": start, "validation_days": len(prediction),
                    "validation_log_score": prediction.log_score.mean(),
                    "validation_return_log_score": prediction.return_log_score.mean(),
                    "initial_train_days": initial.metadata["n_train"],
                    "limited_states": sum(item["limited_evidence"] for item in states)})
            candidates = [row for row in rows if row["asset"] == asset and row["cohort"] == cohort]
            winner = max(candidates, key=lambda row: row["validation_log_score"])
            choices[asset][cohort] = int(winner["k"])
            pd.DataFrame(rows).to_csv(folder / "cohort_selection.csv", index=False)
    save_json(folder / "cohort_selection.json", {"choices": choices,
        "input_fingerprint": input_fingerprint(), "final_test_opened": False,
        "status": "Exploratory extensions; no transition-parameter recovery claim"})
    return choices
