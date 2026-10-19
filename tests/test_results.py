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

    assert state_label("BTCUSDT", "HMM", 3) == "R4 · Biến động rất cao"
    assert state_label("ETHUSDT", "HMM", 3) == "R4 · Thiên giảm, biến động cao"
    assert state_label("ETHUSDT", "HMM", 4) == "R5 · Biến động rất cao"
    for model in ("GMM", "HMM_return", "HMM_B", "TVTP_MACRO", "MSGARCH"):
        assert state_label("BTCUSDT", model, 0) == "R1"
    assert state_label("ETHUSDT", "HDBSCAN_DENSITY", -1) == "Noise"
