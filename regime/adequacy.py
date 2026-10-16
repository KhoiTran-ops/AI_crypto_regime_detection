"""Pre-test sample adequacy: recovery simulations, learning curves and nested origins."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .data import ROOT, config, save_json
from .models import fit_hmm
from .research import dataset, load_selection, state_diagnostics


def recovery(truth, sample_size, repetitions=100, heavy_tail=False):
    from concurrent.futures import ProcessPoolExecutor
    arguments = [(truth, sample_size, replicate, heavy_tail) for replicate in range(repetitions)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        table = pd.DataFrame(pool.map(_recover_one, arguments))
    rate = table.converged.mean()
    return table, {"n_days": sample_size, "repetitions": repetitions, "heavy_tail": heavy_tail,
                   "estimator_starts": len(config()["seeds"]),
                   "convergence_rate": rate, "convergence_mcse": np.sqrt(rate * (1-rate)/repetitions),
                   "transition_rmse": table.get("transition_rmse", pd.Series(dtype=float)).mean(),
                   "volatility_relative_rmse": table.get("volatility_relative_rmse",
                                                        pd.Series(dtype=float)).mean()}


def _recover_one(arguments):
    truth, sample_size, replicate, heavy_tail = arguments
    k, d = truth.model.n_components, len(truth.columns)
    rng = np.random.default_rng(90_000 + replicate)
    x, states = truth.model.sample(sample_size, random_state=90_000 + replicate)
    if heavy_tail:
        x = truth.model.means_[states] + rng.standard_t(5, size=(sample_size, d)) * np.sqrt(
            truth.variances[states] * 3 / 5)
    raw = x * truth.scaler.scale_ + truth.scaler.mean_
    simulated = pd.DataFrame(raw, columns=truth.columns,
                             index=pd.date_range("2000-01-01", periods=sample_size, tz="UTC"))
    try:
        fitted = fit_hmm(simulated, truth.columns, k, config()["seeds"], reference=truth)
        transition_error = np.sqrt(np.mean((fitted.transition - truth.transition) ** 2))
        true_sd = np.sqrt(truth.variances[truth.order, 0]) * truth.scaler.scale_[0]
        fitted_sd = np.sqrt(fitted.variances[fitted.order, 0]) * fitted.scaler.scale_[0]
        volatility_error = np.sqrt(np.mean(((fitted_sd - true_sd) / true_sd) ** 2))
        estimates = {}
        profiles = fitted.raw_profiles()[fitted.order]
        for state in range(k):
            persistence = fitted.transition[state, state]
            estimates.update({f"persistence_R{state+1}": persistence,
                f"duration_R{state+1}": 1/(1-persistence) if persistence < 1 else np.nan,
                f"mean_return_R{state+1}": profiles[state, 0],
                f"volatility_R{state+1}": profiles[state, d] * np.sqrt(365)})
        return {"replicate": replicate, "converged": True, "visited_states": len(np.unique(states)),
                "transition_rmse": transition_error, "volatility_relative_rmse": volatility_error, **estimates}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {"replicate": replicate, "converged": False, "error": str(error)[:160]}


def nested_origins(frame, columns):
    records = []
    for year in range(2020, 2025):
        origin = pd.Timestamp(f"{year}-01-01", tz="UTC")
        inner_start = pd.Timestamp(f"{year-1}-01-01", tz="UTC")
        inner_train = frame.loc[frame.index < inner_start].dropna(subset=columns)
        inner_validation = frame.loc[(frame.index >= inner_start) & (frame.index < origin)].dropna(subset=columns)
        if len(inner_train) < 365:
            continue
        candidates = []
        for k in (2, 3, 4, 5):
            try:
                fitted = fit_hmm(inner_train, columns, k, [11, 23, 37])
                _, alpha = fitted.filter(inner_train)
                prediction, _ = fitted.filter(inner_validation, alpha)
                candidates.append((prediction.log_score.mean(), k))
            except ValueError:
                continue
        if not candidates:
            raise ValueError(f"No estimable model in inner fold {year}")
        chosen = max(candidates)[1]
        outer_train = frame.loc[frame.index < origin].dropna(subset=columns)
        outer_eval = frame.loc[(frame.index >= origin) &
                               (frame.index < pd.Timestamp(f"{year+1}-01-01", tz="UTC"))].dropna(subset=columns)
        fitted = fit_hmm(outer_train, columns, chosen, [11, 23, 37])
        _, alpha = fitted.filter(outer_train)
        prediction, _ = fitted.filter(outer_eval, alpha)
        records.append({"outer_year": year, "inner_validation_year": year-1,
                        "chosen_k_from_past_only": chosen, "n_outer_train": len(outer_train),
                        "n_outer_evaluation": len(outer_eval), "outer_log_score": prediction.log_score.mean()})
        print(f"Nested origin {year}: K={chosen}", flush=True)
    return records


def adequacy():
    selected, cfg = load_selection(), config()
    folder = ROOT / "outputs/research/adequacy"
    folder.mkdir(parents=True, exist_ok=True)
    # Project precision targets, declared before simulations, not universal HMM standards.
    protocol = {"gaussian_replicates": 100, "estimator": "same 20-start estimator as initial training", "transition_rmse_max": .05,
                "volatility_relative_rmse_max": .20, "convergence_rate_min": .90,
                "selection_rule": "Highest validation score among candidates passing recovery at actual n",
                "final_test_used": False, "simulation_limit": "Conditional on fitted Gaussian DGP, not proof of true regimes"}
    save_json(folder / "protocol.json", protocol)
    simulations, curves, decisions = [], [], {}
    scores = pd.read_csv(ROOT / "outputs/research/model_selection.csv")
    for asset in cfg["assets"]:
        frame, columns = dataset(asset)
        train = frame.loc[:cfg["train_end"]].dropna(subset=columns)
        ranking = scores.loc[scores.asset.eq(asset)].sort_values("validation_log_score", ascending=False)
        chosen = None
        for row in ranking.itertuples():
            truth = fit_hmm(train, columns, row.k, cfg["seeds"], cfg["max_iter"])
            print(f"Recovery {asset}: K={row.k}, n={len(train)}, 100 replicates", flush=True)
            replicates, summary = recovery(truth, len(train))
            summary.update({"asset": asset, "k": row.k})
            passed = bool(summary["convergence_rate"] >= .90 and summary["transition_rmse"] <= .05
                          and summary["volatility_relative_rmse"] <= .20)
            summary["passes_project_precision_targets"] = passed
            simulations.append(summary)
            replicates.to_csv(folder / f"{asset}_k{row.k}_recovery.csv", index=False)
            pd.DataFrame(simulations).to_csv(folder / "simulation_summary.csv", index=False)
            if passed:
                chosen = row.k
                decisions[asset] = {"k": chosen, "recovery_passed": True,
                                    "best_density_k": int(ranking.iloc[0].k),
                                    "state_diagnostics": state_diagnostics(truth, train)}
                break
        if chosen is None:
            raise ValueError(f"No {asset} candidate met declared adequacy targets; final test remains closed")
        for size in (730, 1095, 1460, len(train)):
            history = train.iloc[-size:]
            models = [fit_hmm(history, columns, chosen, [seed]) for seed in (11, 23, 37)]
            labels = [model.filter(history)[0].state for model in models]
            ari = np.mean([adjusted_rand_score(a, b) for a, b in combinations(labels, 2)])
            model = max(models, key=lambda item: item.metadata["train_ll"])
            _, alpha = model.filter(history)
            check = frame.loc["2023-01-01":"2023-06-30"].dropna(subset=columns)
            prediction, _ = model.filter(check, alpha)
            curves.append({"asset": asset, "k": chosen, "n_train": len(history), "seed_ari": ari,
                           "validation_2023h1_log_score": prediction.log_score.mean(),
                           "limited_states": sum(item["limited_evidence"] for item in state_diagnostics(model, history))})
        pd.DataFrame(curves).to_csv(folder / "learning_curve.csv", index=False)
        heavy, summary = recovery(truth, len(train), repetitions=50, heavy_tail=True)
        summary.update({"asset": asset, "k": chosen, "passes_project_precision_targets": None})
        simulations.append(summary)
        heavy.to_csv(folder / f"{asset}_heavy_tail.csv", index=False)
        pd.DataFrame(simulations).to_csv(folder / "simulation_summary.csv", index=False)
        pd.DataFrame(nested_origins(frame.loc[:cfg["validation_end"]], columns)).to_csv(
            folder / f"{asset}_nested_origins.csv", index=False)
        selected["choices"][asset]["k"] = chosen
        selected["choices"][asset]["adequacy"] = decisions[asset]
    selected["adequacy_protocol"] = protocol
    selected["sample_status"] = "A_recovery_checked; B_C_require_cohort_diagnostics"
    save_json(ROOT / "outputs/research/selection.json", selected)
    save_json(folder / "decisions.json", decisions)
    return decisions
