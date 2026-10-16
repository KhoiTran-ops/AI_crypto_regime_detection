"""Benchmarks use the same monthly calendar and only historical fitting data."""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from threadpoolctl import threadpool_limits


def monthly_blocks(frame, start, end):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    points = sorted(set([start, *pd.date_range(start, end, freq="MS"), end + pd.Timedelta(days=1)]))
    for left, right in zip(points[:-1], points[1:]):
        yield left, frame.loc[frame.index < left], frame.loc[(frame.index >= left) & (frame.index < right)]


def cluster_walk(frame, columns, kind, k, start, end):
    outputs, diagnostics = [], []
    for left, train, evaluation in monthly_blocks(frame.dropna(subset=columns), start, end):
        if evaluation.empty:
            continue
        scaler = StandardScaler().fit(train[columns])
        x, y = scaler.transform(train[columns]), scaler.transform(evaluation[columns])
        with threadpool_limits(limits=1):
            if kind == "GMM":
                model = GaussianMixture(k, covariance_type="diag", n_init=5,
                                        max_iter=300, reg_covar=.001, random_state=11).fit(x)
                probabilities = model.predict_proba(y)
                score = model.score_samples(y) - np.log(scaler.scale_).sum()
                return_variance = model.covariances_[:, 0]
            else:
                model = KMeans(k, n_init=10, random_state=11).fit(x)
                membership = model.predict(y)
                probabilities = np.eye(k)[membership]
                score = np.full(len(y), np.nan)  # K-means has no predictive density.
                return_variance = np.array([np.var(x[model.labels_ == state, 0]) for state in range(k)])
        order = np.argsort(return_variance)
        probabilities = probabilities[:, order]
        output = pd.DataFrame(probabilities, index=evaluation.index, columns=[f"p_{i}" for i in range(k)])
        output["state"] = probabilities.argmax(axis=1)
        output["p_high_vol"] = probabilities[:, -1]
        output["log_score"] = score
        output["fit_cutoff"] = train.index.max()
        outputs.append(output)
        diagnostics.append({"period": str(left), "n_train": len(train),
                            "converged": bool(getattr(model, "converged_", True))})
    return pd.concat(outputs), diagnostics


def markov_walk(frame, kind, k, start, end, covariates=None):
    columns = ["return"] + (covariates or [])
    source = frame.dropna(subset=columns)
    if len(source) > 1 and (source.index.to_series().diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("Econometric benchmark requires contiguous observations; no silent gap compression")
    outputs, diagnostics, coefficients = [], [], []
    for left, train, evaluation in monthly_blocks(source, start, end):
        if evaluation.empty:
            continue
        combined = pd.concat([train, evaluation])
        exog_train, exog_all = None, None
        if covariates:
            scaler = StandardScaler().fit(train[covariates])
            exog_train = np.column_stack([np.ones(len(train)), scaler.transform(train[covariates])])
            exog_all = np.column_stack([np.ones(len(combined)), scaler.transform(combined[covariates])])
        def construct(data, exog):
            if kind == "MSAR":
                return MarkovAutoregression(data, k_regimes=k, order=1, trend="c", switching_variance=True)
            return MarkovRegression(data, k_regimes=k, trend="c", switching_variance=True, exog_tvtp=exog)
        model = construct(train["return"].to_numpy() * 100, exog_train)
        np.random.seed(11)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            fitted = model.fit(search_reps=5, maxiter=300, disp=False)
            if not fitted.mle_retvals.get("converged", False):
                fitted = model.fit(start_params=fitted.params, method="bfgs", maxiter=700,
                                   em_iter=0, disp=False)
            filtered = construct(combined["return"].to_numpy() * 100, exog_all).filter(fitted.params)
        probabilities = np.asarray(filtered.filtered_marginal_probabilities)
        if probabilities.shape[0] == k and probabilities.shape[1] != k:
            probabilities = probabilities.T
        variances = np.array([fitted.params[model.param_names.index(f"sigma2[{state}]")] for state in range(k)])
        order = np.argsort(variances)
        probabilities = probabilities[-len(evaluation):, order]
        output = pd.DataFrame(probabilities, index=evaluation.index, columns=[f"p_{i}" for i in range(k)])
        output["state"] = probabilities.argmax(axis=1)
        output["p_high_vol"] = probabilities[:, -1]
        output["log_score"] = np.asarray(filtered.llf_obs)[-len(evaluation):] + np.log(100)
        output["fit_cutoff"] = train.index.max()
        outputs.append(output)
        diagnostics.append({"period": str(left), "converged": bool(fitted.mle_retvals.get("converged", False)),
                            "warnings": sorted(set(str(item.message) for item in captured))[:5],
                            "raw_high_vol_state": int(order[-1]), "raw_state_order": order.tolist()})
        for name, value, se in zip(model.param_names, fitted.params, fitted.bse):
            coefficients.append({"period": str(left), "parameter": name, "estimate": float(value),
                                 "std_error": float(se), "ci_low": float(value - 1.96*se),
                                 "ci_high": float(value + 1.96*se)})
    return pd.concat(outputs), diagnostics, pd.DataFrame(coefficients)
