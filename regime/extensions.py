"""Supplemental experiments, isolated from the original protocol and paper generation."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .advanced import bocpd_walk, hdbscan_walk, msgarch_walk
from .benchmarks import markov_walk
from .data import ROOT, config, save_json
from .evaluation import portfolio_comparison
from .external import DEST, build_external
from .final import mean_difference_ci

GROUPS = {
    "MACRO": ["log_vix", "dxy_last_session_return", "ust10_change", "sp500_last_session_return"],
    "CRYPTO": ["stablecoin_cap_growth", "defi_tvl_growth", "market_cap_growth"],
    "LIQUIDITY": ["log_illiquidity30", "btc_eth_corr30"],
}


def build_frames(panel):
    cfg = config()
    frames = {asset: pd.read_parquet(ROOT / f"results/{asset}_market.parquet") for asset in cfg["assets"]}
    for market in frames.values():
        market.index = pd.DatetimeIndex(market.index).as_unit("ns")
    correlation = frames[cfg["assets"][0]]["return"].rolling(30).corr(frames[cfg["assets"][1]]["return"])
    audit, profiles = [], []
    for asset, frame in frames.items():
        frame = frame.join(panel)
        frame["btc_eth_corr30"] = correlation
        # Simple return / USDT quote notional: daily Amihud-type price-impact proxy, not spread.
        frame["amihud"] = np.expm1(frame["return"]).abs() / frame.quote_volume.where(frame.quote_volume > 0)
        frame["log_illiquidity30"] = np.log(frame.amihud.rolling(30).mean().where(lambda value: value > 0))
        frame["market_cap_growth"] = frame[f"{asset[:3].lower()}_market_cap_growth"]
        for covariates in GROUPS.values():
            for column in covariates:
                frame[column + "_lag2"] = frame[column].shift(cfg["execution_delay_days"])
        # Publication sensitivity: one extra weekly reporting cycle, not a fitted optimal lag.
        for column in GROUPS["MACRO"]:
            frame[column + "_delay7_lag2"] = frame[column + "_lag2"].shift(7)
        frames[asset] = frame
        for group, columns in GROUPS.items():
            clean = frame.dropna(subset=[column+"_lag2" for column in columns] + ["return"])
            for split, start, end in [("train", cfg["analysis_start"], cfg["train_end"]),
                                     ("validation", cfg["validation_start"], cfg["validation_end"]),
                                     ("test", cfg["test_start"], cfg["test_end"])]:
                period = clean.loc[start:end]
                audit.append({"asset": asset, "group": group, "split": split, "days": len(period),
                              "first_date": period.index.min(), "last_date": period.index.max()})
        base = pd.read_parquet(ROOT / f"results/{asset}_HMM.parquet")
        base.index = pd.DatetimeIndex(base.index).as_unit("ns")
        joined = frame[["amihud", "btc_eth_corr30", "quote_volume"]].join(base.state, how="inner")
        for state, subset in joined.groupby("state"):
            profiles.append({"asset": asset, "state": state, "days": len(subset),
                "mean_amihud": subset.amihud.mean(), "median_amihud": subset.amihud.median(),
                "mean_btc_eth_corr30": subset.btc_eth_corr30.mean(), "mean_quote_volume": subset.quote_volume.mean()})
        frame.to_parquet(DEST / f"{asset}_features.parquet")
    pd.DataFrame(audit).to_csv(DEST / "sample_counts.csv", index=False)
    pd.DataFrame(profiles).to_csv(DEST / "liquidity_profiles.csv", index=False)
    return frames


def _worker(job):
    asset, kind, parameter, fingerprint = job
    cfg = config()
    label = kind + ("_" + str(parameter) if kind == "HDBSCAN" else "")
    target = DEST / f"{asset}_{label}.parquet"
    receipt = target.with_suffix(".json")
    if target.exists() and receipt.exists():
        prior = json.loads(receipt.read_text(encoding="utf-8"))
        if prior.get("fingerprint") == fingerprint and prior.get("sha256") == hashlib.sha256(target.read_bytes()).hexdigest():
            return asset, label
    frame = pd.read_parquet(DEST / f"{asset}_features.parquet")
    validation, diagnostics, coefficients, decision = None, [], None, {}
    with threadpool_limits(limits=1):
        if kind.startswith(("TVTP_", "MS0_")):
            group = kind.split("_", 1)[1]
            covariates = ([column+"_delay7_lag2" for column in GROUPS["MACRO"]] if group == "MACRO_DELAY7"
                          else [column+"_lag2" for column in GROUPS[group]])
            source = frame.dropna(subset=["return", *covariates])
            active = covariates if kind.startswith("TVTP_") else None
            validation, vd, _ = markov_walk(source.loc[:cfg["validation_end"]], "NHHMM" if active else "MS0", 2,
                cfg["validation_start"], cfg["validation_end"], covariates=active)
            prediction, diagnostics, coefficients = markov_walk(source, "NHHMM" if active else "MS0", 2,
                cfg["test_start"], cfg["test_end"], covariates=active)
            decision = {"covariates": active, "train_first": source.index.min(), "validation_nonconverged": sum(not d["converged"] for d in vd)}
        elif kind == "MSGARCH":
            validation, vd = msgarch_walk(frame.loc[:cfg["validation_end"]], cfg["validation_start"], cfg["validation_end"])
            prediction, diagnostics = msgarch_walk(frame, cfg["test_start"], cfg["test_end"])
            decision = {"specification": "Two-state independent parallel GARCH(1,1), common empirical training mean", "validation_nonconverged": sum(not d["converged"] for d in vd)}
        elif kind == "BOCPD":
            calibration = json.loads((ROOT / "data/snapshots/extension_calibration.json").read_text(encoding="utf-8"))[asset]
            candidates = []
            for hazard in calibration["hazard_candidates"]:
                for weight in (1/calibration["n_train"], 1.):
                    value = bocpd_walk(frame.loc[:cfg["validation_end"]], cfg["validation_start"], cfg["validation_end"], hazard, weight)
                    candidates.append({"hazard": hazard, "prior_weight": weight, "validation_log_score": value.log_score.mean()})
            pd.DataFrame(candidates).to_csv(DEST / f"{asset}_BOCPD_selection.csv", index=False)
            decision = max(candidates, key=lambda row: row["validation_log_score"])
            validation = bocpd_walk(frame.loc[:cfg["validation_end"]], cfg["validation_start"], cfg["validation_end"], decision["hazard"], decision["prior_weight"])
            prediction = bocpd_walk(frame, cfg["test_start"], cfg["test_end"], decision["hazard"], decision["prior_weight"])
        else:
            columns = cfg["core_features"]
            validation, vd = hdbscan_walk(frame.loc[:cfg["validation_end"]], columns, cfg["validation_start"], cfg["validation_end"], parameter)
            prediction, diagnostics = hdbscan_walk(frame, columns, cfg["test_start"], cfg["test_end"], parameter)
            decision = {"min_cluster_size": parameter, "min_samples": len(columns)+1,
                        "selection": "Both empirical train occupancy endpoints reported; no winner selected on test",
                        "validation_noise_fraction": validation.noise.mean(),
                        "validation_membership_strength": validation.membership_strength.mean()}
        if validation is not None:
            validation.to_parquet(DEST / f"{asset}_{label}_validation.parquet")
            if "log_score" in validation:
                decision["validation_log_score"] = validation.log_score.mean()
        prediction.to_parquet(target)
        if coefficients is not None:
            coefficients.to_csv(DEST / f"{asset}_{label}_coefficients.csv", index=False)
        save_json(receipt, {"fingerprint": fingerprint, "decision": decision, "diagnostics": diagnostics,
                           "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    return asset, label


def summarize(completed):
    cfg = config()
    scores, portfolios, uncertainty, comparisons = [], [], [], []
    for asset, label in completed:
        prediction = pd.read_parquet(DEST / f"{asset}_{label}.parquet")
        record = json.loads((DEST / f"{asset}_{label}.json").read_text(encoding="utf-8"))
        row = {"asset": asset, "model": label, "test_days": len(prediction),
               "validation_log_score": record["decision"].get("validation_log_score"),
               "mean_return_log_score": prediction.log_score.mean() if "log_score" in prediction else None,
               "nonconverged_refits": sum(not d.get("converged", True) for d in record["diagnostics"])}
        if "noise" in prediction:
            row.update(noise_fraction=prediction.noise.mean(), mean_membership_strength=prediction.membership_strength.mean())
        scores.append(row)
        if "p_high_vol" in prediction:
            frame = pd.read_parquet(DEST / f"{asset}_features.parquet")
            table, paths, interval = portfolio_comparison(frame, prediction, cfg["test_start"], cfg["test_end"],
                costs=cfg["cost_bps"], target=cfg["volatility_target"])
            table["asset"], table["model"] = asset, label
            portfolios.append(table)
            uncertainty.append({"asset": asset, "model": label, **interval})
            for strategy, path in paths.items():
                path.to_parquet(DEST / f"{asset}_{label}_{strategy}_portfolio.parquet")
        if label.startswith("TVTP_"):
            reference = "MS0_" + label.split("_", 1)[1]
        elif label in ("MSGARCH", "BOCPD"):
            reference = "HMM_return"
        else:
            continue
        folder = DEST if reference.startswith("MS0_") else ROOT / "results"
        other = pd.read_parquet(folder / f"{asset}_{reference}.parquet")
        comparisons.append({"asset": asset, "model": label, "reference": reference,
                            **mean_difference_ci(prediction.log_score, other.log_score)})
    pd.DataFrame(scores).sort_values(["asset", "model"]).to_csv(DEST / "model_scores.csv", index=False)
    pd.concat(portfolios, ignore_index=True).to_csv(DEST / "portfolio_performance.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(DEST / "portfolio_uncertainty.csv", index=False)
    pd.DataFrame(comparisons).to_csv(DEST / "score_comparisons.csv", index=False)


def run_extensions():
    cfg = config()
    before = json.loads((ROOT / "docs/extension_baseline_hashes.json").read_text(encoding="utf-8"))
    for name, expected in before.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Baseline changed: " + name)
    panel = build_external()
    build_frames(panel)
    scientific_paths = [ROOT / f"regime/{name}.py" for name in ("external", "advanced", "extensions", "benchmarks", "evaluation")]
    inputs = [ROOT / "config.json", ROOT / "requirements.lock", *scientific_paths,
              ROOT / "data/snapshots/external_sources.json", ROOT / "data/snapshots/extension_calibration.json",
              *[ROOT / f"results/{asset}_market.parquet" for asset in cfg["assets"]]]
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    save_json(DEST / "protocol.json", {"fingerprint": fingerprint, "hashes": hashes, "config": cfg,
        "status": "Retrospective extension; existing holdout already observed; split and recovery gate unchanged",
        "availability": "Next US business day / UTC day assumption, plus lag2; MACRO_DELAY7 sensitivity",
        "groups": GROUPS, "no_pdf_generated": True})
    calibration = json.loads((ROOT / "data/snapshots/extension_calibration.json").read_text(encoding="utf-8"))
    jobs = []
    for asset in cfg["assets"]:
        jobs += [(asset, prefix + group, None, fingerprint)
                 for group in [*GROUPS, "MACRO_DELAY7"] for prefix in ("TVTP_", "MS0_")]
        jobs += [(asset, "HDBSCAN", size, fingerprint) for size in calibration[asset]["cluster_size_candidates"]]
        jobs += [(asset, name, None, fingerprint) for name in ("BOCPD", "MSGARCH")]
    completed = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(_worker, job): job for job in jobs}
        for future in as_completed(pending):
            asset, label = future.result()
            completed.append((asset, label))
            print(f"Completed {len(completed)}/{len(jobs)}: {asset} {label}", flush=True)
    summarize(sorted(completed))
    save_json(DEST / "run.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "fingerprint": fingerprint, "models": len(completed), "baseline_files_unchanged": len(before),
              "status": "Complete retrospective extension; no PDF"})
