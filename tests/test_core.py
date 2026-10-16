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

def test_forward_filter_matches_enumerated_paths():
    from itertools import product

    from regime.models import forward_filter
    emission = np.array([[.8, .2], [.3, .7], [.6, .4]])
    transition = np.array([[.9, .1], [.2, .8]])
    initial = np.array([.4, .6])
    probabilities, scores = forward_filter(np.log(emission), transition, initial)
    for end in range(1, 4):
        expected = np.zeros(2)
        for path in product(range(2), repeat=end):
            mass = initial[path[0]] * emission[0, path[0]]
            for step in range(1, end):
                mass *= transition[path[step-1], path[step]] * emission[step, path[step]]
            expected[path[-1]] += mass
        assert np.allclose(probabilities[end-1], expected / expected.sum())
    assert np.isfinite(scores).all()


def test_sequence_lengths_handle_pandas_microseconds():
    from regime.models import sequence_lengths
    index = pd.date_range('2020-01-01', periods=4, tz='UTC', unit='us')
    assert sequence_lengths(index) == [4]
    assert sequence_lengths(index.delete(2)) == [2, 1]

def test_oi_uses_valid_fresh_snapshot_and_reports_bad_rows(tmp_path, monkeypatch):
    from regime import derivatives
    monkeypatch.setattr(derivatives, 'ROOT', tmp_path)
    monkeypatch.setattr(derivatives, 'archive', lambda key: {'file': 'unused.zip'})
    frame = pd.DataFrame({'create_time': ['2024-01-01 00:00:00', '2024-01-01 23:55:00'],
                          'symbol': ['BTCUSDT']*2, 'sum_open_interest': [0, 10],
                          'sum_open_interest_value': [0, 100]})
    monkeypatch.setattr(derivatives, 'read_archive', lambda path: frame)
    result = derivatives.oi_day('BTCUSDT', '2024-01-01')
    assert result['oi'] == 10 and result['invalid_snapshots'] == 1

def test_backtest_delays_execution_and_charges_actual_trades():
    from regime.evaluation import backtest
    index = pd.date_range('2025-01-01', periods=5, tz='UTC')
    prices = pd.Series([100., 100., 200., 200., 200.], index=index)
    signal = pd.Series(1., index=index)
    result = backtest(prices, signal, cost_bps=10, delay=2)
    assert result.weight.iloc[:2].eq(0).all()
    assert np.isclose(result.equity.iloc[-1], .999 / 1.001)
    assert result.turnover.sum() > 1.9
