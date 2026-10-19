"""Independent MS-GARCH, conjugate BOCPD, and frozen-tree HDBSCAN benchmarks."""
import warnings

import hdbscan
import numpy as np
import pandas as pd
from hmmlearn import _hmmc
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit, gammaln, logsumexp, softmax
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .benchmarks import monthly_blocks


def garch_filter(parameters, values):
    """Haas et al. parallel GARCH variances; common demeaned return, two Markov states."""
    unconditional = np.exp(parameters[:2])
    weights = softmax(np.column_stack([parameters[2:4], parameters[4:6], np.zeros(2)]), axis=1)
    alpha, beta = weights[:, 0], weights[:, 1]
    omega = unconditional * weights[:, 2]
    stay = expit(parameters[6:8])
    transition = np.array([[stay[0], 1-stay[0]], [1-stay[1], stay[1]]])
    initial = np.array([1-stay[1], 1-stay[0]])
    initial /= initial.sum()
    variance = np.empty((len(values), 2))
    for state in range(2):
        previous_square = np.r_[unconditional[state], values[:-1] ** 2]
        variance[:, state] = lfilter([1.], [1., -beta[state]],
            omega[state] + alpha[state] * previous_square, zi=[beta[state]*unconditional[state]])[0]
    emission = -.5 * (np.log(2*np.pi*variance) + values[:, None]**2/variance)
    # Pinned hmmlearn compiled recursion; agreement with the transparent reference is regression-tested.
    likelihood, lattice = _hmmc.forward_log(initial, transition, emission)
    cumulative = logsumexp(lattice, axis=1)
    probabilities = np.exp(lattice - cumulative[:, None])
    scores = np.diff(np.r_[0., cumulative])
    return likelihood, emission, transition, initial, probabilities, scores


def msgarch_walk(frame, start, end):
    outputs, diagnostics, previous_parameters = [], [], None
    for left, train, evaluation in monthly_blocks(frame.dropna(subset=["return"]), start, end):
        if evaluation.empty:
            continue
        mean, scale = train["return"].mean(), train["return"].std(ddof=1)
        x = ((train["return"]-mean)/scale).to_numpy()
        starts = [np.r_[np.log([.5, 1.5]), [-2., -2.], [2., 2.], [3., 3.]],
                  np.r_[np.log([.25, 2.]), [-1., -1.], [1., 1.], [2., 2.]]]
        if previous_parameters is not None:
            starts.insert(0, previous_parameters)
        def objective(theta):
            value = garch_filter(theta, x)[0]
            return -value/len(x) if np.isfinite(value) else np.finfo(float).max
        fits = [minimize(objective, point, method="L-BFGS-B", bounds=[(-20, 20)]*8,
                         options={"maxiter": 300, "ftol": 1e-10}) for point in starts]
        successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
        if not successful:
            raise ValueError(f"MS-GARCH did not converge at {left}: {[f.message for f in fits]}")
        best = min(successful, key=lambda fit: fit.fun)
        previous_parameters = best.x
        values = ((pd.concat([train, evaluation])["return"]-mean)/scale).to_numpy()
        _, _, transition, _, probabilities, scores = garch_filter(best.x, values)
        high = int(np.argmax(best.x[:2]))
        output = pd.DataFrame(probabilities[-len(evaluation):], index=evaluation.index, columns=["p_0", "p_1"])
        output["p_high_vol"] = output[f"p_{high}"]
        output["next_high_vol"] = probabilities[-len(evaluation):] @ transition[:, high]
        output["state"] = probabilities[-len(evaluation):].argmax(axis=1)
        output["log_score"] = scores[-len(evaluation):] - np.log(scale)
        output["fit_cutoff"] = train.index.max()
        outputs.append(output)
        diagnostics.append({"period": str(left), "converged": bool(best.success),
                            "iterations": int(best.nit), "train_days": len(train), "mean": mean,
                            "scale": scale, "parameters": best.x.tolist(), "transition": transition.tolist(),
                            "high_unconditional_variance_state": high})
    return pd.concat(outputs), diagnostics


def bocpd(values, hazard, mean, variance, prior_weight):
    """Reset before the current observation; exact run-length recursion, no truncation threshold."""
    if not 0 < hazard < 1 or variance <= 0 or prior_weight <= 0:
        raise ValueError("Invalid BOCPD prior")
    # alpha=2 is the smallest integer shape with finite inverse-gamma mean;
    # beta=variance gives E[sigma²]=variance. Prior weight is chosen on validation.
    probability = np.array([1.])
    mu, kappa, alpha, beta = (np.array([value]) for value in (mean, prior_weight, 2., variance))
    def predictive(x, m, k, a, b):
        df, scale2 = 2*a, b*(k+1)/(a*k)
        return gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*np.pi*scale2) - (df+1)/2*np.log1p((x-m)**2/(df*scale2))
    rows = []
    for value in np.asarray(values):
        base = float(predictive(value, mean, prior_weight, 2., variance))
        growth = predictive(value, mu, kappa, alpha, beta)
        joint = np.r_[np.log(hazard)+base,
                      np.log1p(-hazard)+np.log(np.maximum(probability, np.finfo(float).tiny))+growth]
        score = logsumexp(joint)
        probability = np.exp(joint-score)
        rows.append({"log_score": score, "change_probability": probability[0],
                     "map_run_length": int(probability.argmax()), "probability_mass": probability.sum()})
        new_k = kappa+1
        new_mu = (kappa*mu+value)/new_k
        new_beta = beta + kappa*(value-mu)**2/(2*new_k)
        kappa = np.r_[prior_weight+1, new_k]
        mu = np.r_[(prior_weight*mean+value)/(prior_weight+1), new_mu]
        alpha = np.r_[2.5, alpha+.5]
        beta = np.r_[variance+prior_weight*(value-mean)**2/(2*(prior_weight+1)), new_beta]
    return pd.DataFrame(rows)


def bocpd_walk(frame, start, end, hazard, prior_weight):
    start_date, end_date = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    train = frame.loc[frame.index < start_date].dropna(subset=["return"])
    combined = frame.loc[frame.index <= end_date].dropna(subset=["return"])
    result = bocpd(combined["return"].to_numpy(), hazard, train["return"].mean(),
                   train["return"].var(ddof=1), prior_weight)
    result.index = combined.index
    result = result.loc[start_date:].copy()
    result["fit_cutoff"] = train.index.max()
    return result


def hdbscan_walk(frame, columns, start, end, min_cluster_size):
    outputs, diagnostics = [], []
    for left, train, evaluation in monthly_blocks(frame.dropna(subset=columns), start, end):
        if evaluation.empty:
            continue
        scaler = StandardScaler().fit(train[columns])
        x, y = scaler.transform(train[columns]), scaler.transform(evaluation[columns])
        # min_samples=dimension+1 is the minimum full-dimensional simplex;
        # tested separately from min_cluster_size, as recommended in official docs.
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            fitted = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=len(columns)+1,
                prediction_data=True, gen_min_span_tree=True).fit(x)
            labels, strength = hdbscan.approximate_predict(fitted, y)
        states = sorted(set(fitted.labels_) - {-1})
        rank = sorted(states, key=lambda state: train["return"].to_numpy()[fitted.labels_ == state].var())
        mapping = {state: order for order, state in enumerate(rank)}
        output = pd.DataFrame({"state": [mapping.get(label, -1) for label in labels],
                               "membership_strength": strength, "noise": labels == -1}, index=evaluation.index)
        output["fit_cutoff"] = train.index.max()
        outputs.append(output)
        diagnostics.append({"period": str(left), "clusters": len(states),
            "train_noise_fraction": float(np.mean(fitted.labels_ == -1)),
            "validation_or_test_noise_fraction": float(np.mean(labels == -1)),
            "cluster_persistence": fitted.cluster_persistence_.tolist(),
            "warnings": sorted({str(item.message) for item in caught})})
    return pd.concat(outputs), diagnostics
