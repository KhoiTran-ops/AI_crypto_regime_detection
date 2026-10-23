"""Byte-exact packaging, deduplication and safe replay of published findings."""
import pytest

from regime.results import Results, materialized_results, pack_results


def test_round_trip_dedup_and_corruption(tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    for name in ("first.json", "second.json"):
        (folder / name).write_bytes(b'{"answer":42}')
    pack_results(tmp_path)
    first_archive = (folder / "research.zip").read_bytes()
    pack_results(tmp_path)
    assert (folder / "research.zip").read_bytes() == first_archive
    store = Results(tmp_path)
    assert store.json("first.json") == {"answer": 42}
    assert store.index["first.json"] == store.index["second.json"]
    assert not (folder / "first.json").exists()
    with materialized_results(tmp_path):
        assert (folder / "second.json").read_bytes() == b'{"answer":42}'
        (folder / "new.json").write_bytes(b'{}')
    assert Results(tmp_path).json("new.json") == {}
    (folder / "first.json").write_bytes(b'changed')
    with pytest.raises(ValueError, match="changed"):
        with materialized_results(tmp_path):
            pass


def test_primary_regime_labels_do_not_leak_across_models():
    from regime.labels import state_label

    assert state_label("BTCUSDT", "HMM", 3) == "R4 · High-volatility regime"
    assert state_label("ETHUSDT", "HMM", 3) == "R4 · Bearish regime — high volatility"
    assert state_label("ETHUSDT", "HMM", 4) == "R5 · High-volatility regime"
    for model in ("GMM", "HMM_return", "HMM_B", "TVTP_MACRO"):
        assert state_label("BTCUSDT", model, 0) == "R1"
    assert state_label("BTCUSDT", "MSGARCH", 0) == "S1 raw"
    assert state_label("ETHUSDT", "HDBSCAN_DENSITY", -1) == "Noise · Unclassified observations"


def test_reproduce_initializes_workdir_on_clean_checkout(tmp_path, monkeypatch):
    from regime import derivatives, reproduce

    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    monkeypatch.setattr(reproduce, "verify", lambda: None)
    monkeypatch.setattr(reproduce, "config", lambda: {"assets": []})

    class ReplayReached(Exception):
        pass

    def stop_at_replay():
        raise ReplayReached

    monkeypatch.setattr(derivatives, "funding", stop_at_replay)
    with pytest.raises(ReplayReached):
        reproduce.reproduce()


def test_labels_follow_each_fit_without_changing_predictions():
    import pandas as pd

    from regime.labels import label_predictions, state_label

    assert "Low-volatility" in state_label("ETHUSDT", "GMM", 0, k=5)
    assert "Intermediate-volatility" in state_label("BTCUSDT", "HMM_B", 1, k=3)
    assert "High-volatility" in state_label("BTCUSDT", "TVTP_MACRO", 1, k=2)
    index = pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True)
    value = pd.DataFrame({"state": [1, 1]}, index=index)
    fits = [{"period": str(index[0]), "clusters": 2}, {"period": str(index[1]), "clusters": 3}]
    labeled = label_predictions("BTCUSDT", "HDBSCAN_DENSITY", value, fits)
    assert "High-volatility" in labeled.market_regime.iloc[0]
    assert "Intermediate-volatility" in labeled.market_regime.iloc[1]
    fits = [{"period": str(index[0]), "high_unconditional_variance_state": 1},
            {"period": str(index[1]), "high_unconditional_variance_state": 0}]
    labeled = label_predictions("ETHUSDT", "MSGARCH", value, fits)
    assert "High-volatility" in labeled.market_regime.iloc[0]
    assert "Low-volatility" in labeled.market_regime.iloc[1]
    pd.testing.assert_frame_equal(labeled[value.columns], value)
    assert list(value) == ["state"]


def test_dashboard_appends_live_history_without_replacing_published_rows():
    import pandas as pd

    from regime.dashboard_data import append_history

    dates=pd.date_range("2026-09-29",periods=4,tz="UTC")
    historical=pd.DataFrame({"value":[1.,2.]},index=dates[:2])
    live=pd.DataFrame({"value":[999.,3.,4.]},index=dates[1:])
    result=append_history(historical,live)
    pd.testing.assert_frame_equal(result.loc[dates[:2]],historical)
    assert result.value.tolist()==[1.,2.,3.,4.]
    assert not result.index.has_duplicates


def test_dashboard_predictions_include_published_and_live_periods(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from regime import dashboard_data
    from regime.live import Store
    from regime.results import Results

    store=Store(tmp_path / "live.sqlite")
    monkeypatch.setattr(dashboard_data,"runtime",lambda:SimpleNamespace(store=store))
    original=Results().frame("BTCUSDT_HMM.parquet")
    extra=original.tail(1).copy()
    extra.index=extra.index+pd.Timedelta(days=1)
    store.frame("model:BTCUSDT:HMM:predictions",extra)
    result,_=dashboard_data.prediction("BTCUSDT","HMM")
    assert result.index[0]==original.index[0]
    assert result.index[-1]==extra.index[-1]
    assert len(result)==len(original)+1
    pd.testing.assert_frame_equal(result.loc[original.index,original.columns],original)


def test_dashboard_keeps_published_portfolios_when_no_new_predictions(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from regime import dashboard_data
    from regime.live import Store
    from regime.results import Results

    monkeypatch.setattr(dashboard_data,"runtime",lambda:SimpleNamespace(store=Store(tmp_path / "live.sqlite")))
    dashboard_data._portfolio.clear()
    rows,paths=dashboard_data.portfolio_data("BTCUSDT","HMM")
    expected=Results().table("portfolio_performance").query('asset == "BTCUSDT" and model == "HMM"')
    pd.testing.assert_frame_equal(rows,expected)
    for strategy in ["buy_hold","vol_target","regime_aware"]:
        pd.testing.assert_frame_equal(paths[10,strategy],Results().frame(f"BTCUSDT_HMM_{strategy}_portfolio.parquet"))


def test_dashboard_model_comparisons_use_common_observation_dates(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    from regime import dashboard_data
    from regime.live import Store
    from regime.results import Results

    store=Store(tmp_path / "live.sqlite")
    monkeypatch.setattr(dashboard_data,"runtime",lambda:SimpleNamespace(store=store))
    dashboard_data._scores.clear()
    archive=Results()
    for model in ["HMM","GMM"]:
        extra=archive.frame("BTCUSDT_"+model+".parquet").tail(1).copy()
        extra.index=extra.index+pd.Timedelta(days=1)
        store.frame("model:BTCUSDT:"+model+":predictions",extra)
        store.json("model:BTCUSDT:"+model,{"updated_at":1})
    scores=dashboard_data.current_scores("BTCUSDT").set_index("model")
    assert scores.loc["HMM","return_score_end"]=="2026-09-30"
    assert scores.loc["HMM","joint_score_end"]=="2026-10-01"
    original=archive.table("model_scores").query('asset == "BTCUSDT" and model == "HMM"').iloc[0]
    assert scores.loc["HMM","mean_return_log_score"]==pytest.approx(original.mean_return_log_score)


def test_dashboard_shows_background_fetch_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from regime import dashboard_data, dashboard_views
    from regime.live import Store

    store=Store(tmp_path / "live.sqlite")
    for role in ("quotes", "daily"):
        store.json("worker:"+role,{"error":"HTTP 451"})
    service=SimpleNamespace(store=store,threads=[object()])
    monkeypatch.setattr(dashboard_data,"runtime",lambda:service)
    monkeypatch.setattr(dashboard_views,"runtime",lambda:service)
    app=AppTest.from_string("from regime.dashboard_views import data\ndata('BTCUSDT',None)").run(timeout=60)
    assert not app.exception
    assert [error.value for error in app.error]==[
        "Cập nhật giá: HTTP 451", "Cập nhật dữ liệu ngày: HTTP 451"]
    statuses=next(frame.value for frame in app.dataframe if "Trạng thái" in frame.value)
    assert statuses["Trạng thái"].eq("Chưa có cập nhật mới").all()
