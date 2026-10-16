import numpy as np
import pandas as pd
import pytest

from regime.data import normalize_spot
from regime.features import build_features


def test_archive_timestamp_units_and_invalid_ohlc():
    row = [1503014400000, 10, 12, 9, 11, 20, 1503100799999, 220, 5, 10, 110, 0]
    before = normalize_spot(pd.DataFrame([row]), "BTCUSDT")
    row[0], row[6] = 1735689600000000, 1735775999999999
    after = normalize_spot(pd.DataFrame([row]), "BTCUSDT")
    assert str(before.index[0].date()) == "2017-08-18"
    assert str(after.index[0].date()) == "2025-01-01"
    row[2] = 5
    with pytest.raises(ValueError, match="OHLC"):
        normalize_spot(pd.DataFrame([row]), "BTCUSDT")


def test_features_are_prefix_invariant_and_do_not_bridge_gaps():
    index = pd.date_range("2017-08-18", periods=100, tz="UTC")
    frame = pd.DataFrame({"close": np.exp(np.arange(100) * .001),
                          "quote_volume": np.arange(100) + 100.0}, index=index)
    original = build_features(frame)
    changed = frame.copy()
    changed.loc[index[80]:, "close"] *= 3
    pd.testing.assert_frame_equal(original.loc[:index[79]], build_features(changed).loc[:index[79]])
    missing = build_features(frame.drop(index[65]))
    assert pd.isna(missing.loc[index[66], "return"])

def test_funding_daily_calendar_is_utc(tmp_path, monkeypatch):
    import json

    from regime import derivatives
    monkeypatch.setattr(derivatives, 'ROOT', tmp_path)
    monkeypatch.setattr(derivatives, 'config', lambda: {'assets': ['BTCUSDT'], 'end': '2020-01-03'})
    raw = tmp_path / 'data/raw/funding/BTCUSDT.json'
    raw.parent.mkdir(parents=True)
    events = [{'fundingTime': int(day.timestamp()*1000), 'fundingRate': '0.0001'}
              for day in pd.date_range('2020-01-01', '2020-01-03 16:00', freq='8h', tz='UTC')]
    raw.write_text(json.dumps(events))
    derivatives.funding()
    daily = pd.read_parquet(tmp_path / 'data/processed/BTCUSDT_funding.parquet')
    assert len(daily) == 3 and daily.funding_count.eq(3).all()
