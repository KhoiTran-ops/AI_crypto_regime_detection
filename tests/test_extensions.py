"""Focused invariants for supplemental data and causal model recursions."""
import numpy as np
import pandas as pd
import pytest

from regime.external import asof_panel, validate_observations


def test_external_is_unavailable_before_release():
    raw = pd.DataFrame({"series": ["vix", "vix"], "observation_date": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
                        "available_date": pd.to_datetime(["2020-01-03", "2020-01-06"], utc=True), "value": [10., 99.]})
    raw["available_date"] = raw.available_date.dt.as_unit("us")
    panel = asof_panel(raw, pd.date_range("2020-01-01", "2020-01-07", tz="UTC"))
    assert pd.isna(panel.loc["2020-01-02", "vix"])
    assert panel.loc["2020-01-05", "vix"] == 10
    assert panel.loc["2020-01-06", "vix"] == 99
    pd.testing.assert_frame_equal(panel.iloc[:4], asof_panel(raw.iloc[:1], panel.index[:4]))


def test_conflicting_source_observations_rejected():
    row = {"series": "x", "observation_date": pd.Timestamp("2020-01-01", tz="UTC"),
           "available_date": pd.Timestamp("2020-01-02", tz="UTC"), "value": 1.}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_observations(pd.DataFrame([row, row | {"value": 2.}]))


def test_compiled_filter_matches_reference():
    from regime.advanced import garch_filter
    from regime.models import forward_filter
    x = np.random.default_rng(7).normal(size=80)
    _, emission, transition, initial, probabilities, scores = garch_filter(np.zeros(8), x)
    expected, expected_scores = forward_filter(emission, transition, initial)
    np.testing.assert_allclose(probabilities, expected, atol=1e-12)
    np.testing.assert_allclose(scores, expected_scores, atol=1e-12)


def test_bocpd_prefix_and_probability_mass():
    from regime.advanced import bocpd
    x = np.random.default_rng(9).normal(size=60)
    full = bocpd(x, hazard=.05, mean=0., variance=1., prior_weight=1.)
    short = bocpd(x[:40], hazard=.05, mean=0., variance=1., prior_weight=1.)
    pd.testing.assert_frame_equal(full.iloc[:40], short)
    assert full.change_probability.between(0, 1).all()
    assert np.isfinite(full.log_score).all()


def test_garch_variance_recursion_and_change_response():
    from regime.advanced import bocpd, garch_filter
    emission = garch_filter(np.zeros(8), np.zeros(3))[1]
    np.testing.assert_allclose(np.exp(-2*emission[:, 0])/(2*np.pi), [1, 2/3, 5/9])
    rng = np.random.default_rng(14)
    values = np.r_[rng.normal(size=70), rng.normal(10, 1, size=10)]
    result = bocpd(values, hazard=.03, mean=0., variance=1., prior_weight=1.)
    assert result.change_probability.iloc[70] > result.change_probability.iloc[60:70].max()
