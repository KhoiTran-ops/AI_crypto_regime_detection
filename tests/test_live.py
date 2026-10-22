"""Operational guarantees for the live pipeline; no external requests."""
import io
from urllib.error import HTTPError

import pandas as pd
import pytest


@pytest.mark.parametrize("code,retry,until", [(429,"7200",8200),(418,"7200",8200),(418,None,260200)])
def test_rate_limit_survives_restart_and_honors_full_retry_after(tmp_path, code, retry, until):
    from regime.live import Client, Deferred, Store

    calls = []
    def limited(*args, **kwargs):
        calls.append(1)
        raise HTTPError(args[0].full_url, code, "limited", {"Retry-After": retry} if retry else None, io.BytesIO())
    client = Client(Store(tmp_path / "live.sqlite"), opener=limited, clock=lambda: 1000.)
    with pytest.raises(Deferred):
        client.get("https://api.binance.com/api/v3/time")
    restarted = Client(Store(tmp_path / "live.sqlite"), opener=limited, clock=lambda: 1061.)
    with pytest.raises(Deferred):
        restarted.get("https://api.binance.com/api/v3/time")
    assert len(calls) == 1
    assert restarted.store.json("http:api.binance.com")["until"] >= until


def test_only_complete_utc_bars_and_contiguous_append():
    from regime.live_data import closed_bars, merge_days

    date = pd.Timestamp("2026-10-21", tz="UTC")
    ms = int(date.timestamp() * 1000)
    row = [ms, "100", "110", "90", "105", "1", ms+86400000-1, "105", 1, "0.5", "52", "0"]
    rows = [row, [ms+86400000, *row[1:]]]
    result = closed_bars(rows, "BTCUSDT", date+pd.Timedelta(days=1))
    assert result.index.tolist() == [date]
    assert merge_days(result, result).equals(result)
    changed = result.copy()
    changed.loc[date, "close"] = 106
    with pytest.raises(ValueError, match="Conflicting"):
        merge_days(result, changed)
    missing = result.copy()
    missing.index += pd.Timedelta(days=2)
    with pytest.raises(ValueError, match="gap"):
        merge_days(result, missing)


def test_store_frames_and_process_lock(tmp_path):
    from regime.live import Store, exclusive

    store = Store(tmp_path / "live.sqlite")
    value = pd.DataFrame({"value": [1., 2.]})
    store.frame("sample", value)
    pd.testing.assert_frame_equal(store.frame("sample"), value)
    with exclusive(tmp_path / "worker.lock") as first:
        assert first
        with exclusive(tmp_path / "worker.lock") as second:
            assert not second


def test_quote_poll_is_shared_and_uses_valid_compact_symbols(tmp_path):
    import json
    from urllib.parse import parse_qs, urlsplit

    from regime.live import Store, update_quotes

    calls=[]
    class Provider:
        def json(self, url):
            symbols=parse_qs(urlsplit(url).query)["symbols"][0]
            assert " " not in symbols
            calls.append(url)
            return [{"symbol":symbol,"price":"100"} for symbol in json.loads(symbols)]
    store=Store(tmp_path / "live.sqlite")
    first=update_quotes(store,Provider())
    second=update_quotes(store,Provider())
    assert first==second and len(calls)==1


def test_daily_models_persist_predictions_and_skip_completed_day(tmp_path, monkeypatch):
    import numpy as np

    from regime import live_models
    from regime.live import Store

    store=Store(tmp_path / "live.sqlite")
    dates=pd.date_range("2026-10-01", periods=2, tz="UTC", name="date")
    output=pd.DataFrame({"p_0":[0.2,0.4],"p_1":[0.8,0.6],"state":[1,1],
                         "p_high_vol":[0.8,0.6],"fit_cutoff":pd.Timestamp("2026-09-30",tz="UTC")},index=dates)
    calls=[]
    class Archive:
        def table(self, name):
            return pd.DataFrame({"asset":["BTCUSDT","ETHUSDT"],"model":["HMM_return","KMeans"]})
    def estimate(*args):
        calls.append(1)
        return output.copy(),[]
    monkeypatch.setattr(live_models,"Results",Archive)
    monkeypatch.setattr(live_models,"estimate",estimate)
    for _ in range(2):
        live_models.update_models(store,{"BTCUSDT":pd.DataFrame()},dates[-1],core_only=True)
    assert len(calls)==1
    saved=store.frame("model:BTCUSDT:HMM_return:predictions")
    np.testing.assert_allclose(saved[["p_0","p_1"]],output[["p_0","p_1"]])
    assert store.json("model:BTCUSDT:HMM_return")["through"]=="2026-10-02"


def test_hmm_numeric_cache_preserves_filter_and_state_order(tmp_path):
    import numpy as np
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    from regime.live import Store
    from regime.live_models import load_fit, save_fit
    from regime.models import FittedHMM

    frame=pd.DataFrame({"return":[-0.02,0.01,0.03,-0.01]},index=pd.date_range("2026-10-01",periods=4,tz="UTC"))
    model=GaussianHMM(2,covariance_type="diag")
    model.n_features=1
    model.startprob_=np.array([0.4,0.6])
    model.transmat_=np.array([[0.9,0.1],[0.2,0.8]])
    model.means_=np.array([[-0.5],[0.5]])
    model.covars_=np.array([[0.5],[2.]])
    fitted=FittedHMM(model,StandardScaler().fit(frame),["return"],np.array([1,0]),{})
    store=Store(tmp_path / "live.sqlite")
    save_fit(store,"fit",fitted)
    restored=load_fit(store,"fit")
    pd.testing.assert_frame_equal(fitted.filter(frame)[0],restored.filter(frame)[0])


@pytest.mark.parametrize("stamp,expected", [("2026-10-23T00:04:00Z","2026-10-21"),("2026-10-23T00:06:00Z","2026-10-22")])
def test_daily_cutoff_uses_exchange_utc_clock(tmp_path, monkeypatch, stamp, expected):
    from regime import live_data, live_models
    from regime.live import Store, daily_update

    store=Store(tmp_path / "live.sqlite")
    observed=[]
    class Provider:
        def json(self, url):
            return {"serverTime":int(pd.Timestamp(stamp).timestamp()*1000)}
    def spot(store,client,cutoff):
        observed.append(str(cutoff.date()))
        return {a:pd.DataFrame() for a in ["BTCUSDT","ETHUSDT"]}
    def models(store,frames,cutoff,core_only):
        store.json("models",{"assets":{a:["HMM"] for a in frames}})
        for a in frames:
            store.json("model:"+a+":HMM",{"through":str(cutoff.date())})
    monkeypatch.setattr(live_data,"update_spot",spot)
    monkeypatch.setattr(live_data,"update_controls",lambda store,client,frames,cutoff:frames)
    monkeypatch.setattr(live_models,"update_models",models)
    daily_update(store,Provider())
    assert observed==[expected]
    assert store.json("daily")["complete"]


@pytest.mark.parametrize("fail", [False, True])
def test_fred_attempts_one_batch_per_utc_day_across_restarts(tmp_path, fail):
    from urllib.parse import parse_qs, urlsplit

    from regime.live import Store
    from regime.live_data import fred_daily

    calls=[]
    class Provider:
        def get(self, url):
            calls.append(url)
            if fail:
                raise TimeoutError("timed out")
            series=parse_qs(urlsplit(url).query)["id"][0]
            return f"observation_date,{series}\n2026-10-20,20.5\n".encode()
    for _ in range(2):
        store=Store(tmp_path / "live.sqlite")
        if fail:
            with pytest.raises(RuntimeError,match="timed out"):
                fred_daily(store,Provider(),pd.Timestamp("2026-10-21",tz="UTC"))
        else:
            assert set(fred_daily(store,Provider(),pd.Timestamp("2026-10-21",tz="UTC")).columns)=={
                "observation_date","VIXCLS","DGS10","DGS2","SP500"}
    assert len(calls)==(1 if fail else 4)
    if fail:
        with pytest.raises(RuntimeError):
            fred_daily(store,Provider(),pd.Timestamp("2026-10-22",tz="UTC"))
    else:
        fred_daily(store,Provider(),pd.Timestamp("2026-10-22",tz="UTC"))
    assert len(calls)==(2 if fail else 8)
    # An older cutoff must not reopen a previously attempted day.
    if fail:
        with pytest.raises(RuntimeError):
            fred_daily(store,Provider(),pd.Timestamp("2026-10-21",tz="UTC"))
    else:
        fred_daily(store,Provider(),pd.Timestamp("2026-10-21",tz="UTC"))
    assert len(calls)==(2 if fail else 8)
    if not fail:
        fail=True
        prior=fred_daily(store,Provider(),pd.Timestamp("2026-10-23",tz="UTC"))
        assert "VIXCLS" in prior and store.json("fred_daily")["error"]=="timed out"
        assert store.json("fred_daily")["last_success_cutoff"]=="2026-10-22"
        fred_daily(Store(tmp_path / "live.sqlite"),Provider(),pd.Timestamp("2026-10-23",tz="UTC"))
        assert len(calls)==9
