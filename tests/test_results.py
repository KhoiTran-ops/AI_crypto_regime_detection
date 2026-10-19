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
