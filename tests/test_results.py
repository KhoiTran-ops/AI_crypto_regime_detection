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
