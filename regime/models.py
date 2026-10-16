"""Gaussian HMM fitting and causal forward filtering (never full-sample smoothing)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def forward_filter(log_emission, transition, initial, previous=None, resets=None):
    probabilities = np.empty_like(log_emission)
    scores = np.empty(len(log_emission))
    transition = np.clip(transition, 1e-300, 1)
    transition = transition / transition.sum(axis=1, keepdims=True)
    alpha = previous
    for row in range(len(log_emission)):
        if alpha is None or (resets is not None and resets[row]):
            prior = initial
        else:
            prior = alpha @ transition
        log_alpha = np.log(np.clip(prior, 1e-300, 1)) + log_emission[row]
        scores[row] = logsumexp(log_alpha)
        alpha = np.exp(log_alpha - scores[row])
        probabilities[row] = alpha
    return probabilities, scores


def sequence_lengths(index):
    breaks = np.r_[0, np.flatnonzero(np.diff(index.as_unit("ns").asi8) != pd.Timedelta(days=1).value) + 1, len(index)]
    return np.diff(breaks).tolist()


@dataclass
class FittedHMM:
    model: GaussianHMM
    scaler: StandardScaler
    columns: list
    order: np.ndarray
    metadata: dict

    @property
    def variances(self):
        covariance = self.model.covars_
        return np.diagonal(covariance, axis1=1, axis2=2) if covariance.ndim == 3 else covariance

    @property
    def stress_state(self):
        raw = np.argmax(self.variances[:, self.columns.index("return")])
        return int(np.flatnonzero(self.order == raw)[0])

    @property
    def transition(self):
        return self.model.transmat_[np.ix_(self.order, self.order)]

    def raw_profiles(self):
        means = self.model.means_ * self.scaler.scale_ + self.scaler.mean_
        standard_deviations = np.sqrt(self.variances) * self.scaler.scale_
        return np.column_stack([means, standard_deviations])

    def filter(self, frame, previous=None):
        values = self.scaler.transform(frame[self.columns])
        variance = np.maximum(self.variances, 1e-8)
        log_emission = -.5 * (len(self.columns) * np.log(2 * np.pi) + np.log(variance).sum(axis=1)
                             + ((values[:, None, :] - self.model.means_) ** 2 / variance).sum(axis=2))
        resets = np.r_[False, np.diff(frame.index.as_unit("ns").asi8) != pd.Timedelta(days=1).value]
        probabilities, scores = forward_filter(log_emission, self.model.transmat_, self.model.startprob_,
                                              previous=previous, resets=resets)
        # Return density in original feature units, including the scaler Jacobian.
        scores = scores - np.log(self.scaler.scale_).sum()
        ordered = probabilities[:, self.order]
        output = pd.DataFrame(ordered, index=frame.index, columns=[f"p_{i}" for i in range(len(self.order))])
        output["state"] = ordered.argmax(axis=1)
        output["p_high_vol"] = ordered[:, self.stress_state]
        output["log_score"] = scores
        initial_prior = self.model.startprob_ if previous is None else previous @ self.model.transmat_
        priors = np.vstack([initial_prior, probabilities[:-1] @ self.model.transmat_])
        priors[resets] = self.model.startprob_
        j = self.columns.index("return")
        return_emission = -.5 * (np.log(2 * np.pi * variance[:, j])
            + (values[:, None, j] - self.model.means_[:, j]) ** 2 / variance[:, j])
        output["return_log_score"] = logsumexp(np.log(np.clip(priors, 1e-300, 1)) + return_emission,
                                                axis=1) - np.log(self.scaler.scale_[j])
        raw_stress = self.order[self.stress_state]
        output["next_high_vol"] = probabilities @ self.model.transmat_[:, raw_stress]
        output["entropy"] = -(ordered * np.log(np.clip(ordered, 1e-12, 1))).sum(axis=1)
        return output, probabilities[-1] if len(probabilities) else previous


def fit_hmm(frame, columns, k, seeds, max_iter=300, reference=None):
    clean = frame.dropna(subset=columns)
    scaler = StandardScaler().fit(clean[columns])
    values = scaler.transform(clean[columns])
    lengths = sequence_lengths(clean.index)
    trials, best = [], None
    for seed in seeds:
        model = GaussianHMM(k, covariance_type="diag", n_iter=max_iter, tol=.01, min_covar=.001,
                            random_state=seed, implementation="scaling")
        try:
            with threadpool_limits(limits=1):
                model.fit(values, lengths=lengths)
            score = model.score(values, lengths=lengths)
            history = list(model.monitor_.history)
            delta = history[-1] - history[-2] if len(history) > 1 else np.inf
            converged = bool(np.isfinite(score) and -.001 <= delta < .02)
            trials.append({"seed": seed, "train_ll": score, "iterations": model.monitor_.iter,
                           "converged": converged, "last_delta": delta})
            if converged and (best is None or score > best[0]):
                best = score, model, seed
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            trials.append({"seed": seed, "converged": False, "error": str(error)})
    if best is None:
        raise ValueError(f"No converged HMM: K={k}, n={len(clean)}; trials={trials}")
    score, model, seed = best
    n_parameters = (k - 1) + k * (k - 1) + 2 * k * len(columns)
    fitted = FittedHMM(model, scaler, columns, np.arange(k),
                       {"k": k, "seed": seed, "n_train": len(clean), "train_end": str(clean.index.max()),
                        "n_parameters": n_parameters, "train_ll": score,
                        "aic": -2 * score + 2 * n_parameters,
                        "bic": -2 * score + n_parameters * np.log(len(clean)),
                        "converged_starts": sum(item["converged"] for item in trials), "trials": trials})
    if reference is None:
        fitted.order = np.argsort(fitted.variances[:, columns.index("return")])
    else:
        prior_profiles = reference.raw_profiles()[reference.order]
        new_profiles = fitted.raw_profiles()
        scales = np.r_[reference.scaler.scale_, reference.scaler.scale_]
        distances = (((prior_profiles[:, None] - new_profiles[None, :]) / scales) ** 2).sum(axis=2)
        old, new = linear_sum_assignment(distances)
        fitted.order = new[np.argsort(old)]
    return fitted


def walk_forward(frame, columns, k, start, end, seeds, max_iter=300, freq="MS", initial_seeds=None):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    boundaries = pd.date_range(start, end + pd.Timedelta(days=1), freq=freq)
    boundaries = sorted(set([start, *boundaries, end + pd.Timedelta(days=1)]))
    predictions, fits, reference = [], [], None
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        train = frame.loc[frame.index < left].dropna(subset=columns)
        evaluation = frame.loc[(frame.index >= left) & (frame.index < right)].dropna(subset=columns)
        if evaluation.empty:
            continue
        active_seeds = initial_seeds if reference is None and initial_seeds is not None else seeds
        fitted = fit_hmm(train, columns, k, active_seeds, max_iter, reference)
        _, alpha = fitted.filter(train)
        if (evaluation.index[0] - train.index[-1]).days > 1:
            alpha = None
        result, _ = fitted.filter(evaluation, previous=alpha)
        result["fit_cutoff"] = train.index.max()
        result["model_id"] = f"{left.date()}-k{k}-s{fitted.metadata['seed']}"
        result["stress_state"] = fitted.stress_state
        result["regime_signal"] = 1 - result.p_high_vol
        fits.append({"period_start": str(left), **fitted.metadata,
                     "transition": fitted.transition.tolist(), "stress_state": fitted.stress_state})
        predictions.append(result)
        reference = fitted
    if not predictions:
        raise ValueError("No out-of-sample predictions")
    return pd.concat(predictions), fits, reference
