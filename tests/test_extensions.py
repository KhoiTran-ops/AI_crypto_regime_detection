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
