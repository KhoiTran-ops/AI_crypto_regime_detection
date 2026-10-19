"""Compact, byte-exact result store; scientific writers retain their published format."""
import json
import zipfile
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath

import pandas as pd

from .data import ROOT

SUMMARY_NAMES = ("model_scores", "portfolio_performance", "score_comparisons", "portfolio_uncertainty",
                 "sample_counts", "data_sources", "regime_profiles")


class Results:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.folder = self.root / "results"
        self.archive = self.folder / "research.zip"
        self.index = {}
        if self.archive.exists():
            with zipfile.ZipFile(self.archive) as archive:
                self.index = json.loads(archive.read("index.json"))["files"]
        for name in self.index:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe result path: " + name)

    def exists(self, name):
        return name in self.index or (self.folder / name).is_file()

    def read(self, name):
        path = self.folder / name
        if path.is_file():
            return path.read_bytes()
        with zipfile.ZipFile(self.archive) as archive:
            value = archive.read("objects/" + self.index[name])
        if sha256(value).hexdigest() != self.index[name]:
            raise ValueError("Archived result changed: " + name)
        return value

    def json(self, name):
        return json.loads(self.read(name))

    def frame(self, name):
        data = BytesIO(self.read(name))
        return pd.read_parquet(data) if name.endswith(".parquet") else pd.read_csv(data)

    def table(self, name):
        return pd.read_csv(self.folder / "tables" / (name + ".csv"))

    def model_path(self, asset, model):
        name = f"{asset}_{model}.parquet"
        return "extensions/" + name if self.exists("extensions/" + name) else name


def _summaries(store):
    for name in SUMMARY_NAMES:
        files = [name + ".csv", "extensions/" + name + ".csv"]
        if name == "data_sources":
            files = ["data_quality.csv", "extensions/source_coverage.csv"]
        tables = [store.frame(path) for path in files if store.exists(path)]
        if tables:
            if name == "sample_counts":
                tables = [t.rename(columns={"group": "cohort", "days": "complete_feature_days",
                         "first_date": "start", "last_date": "end"}) for t in tables]
            combined = pd.concat(tables, ignore_index=True, sort=False)
            if name == "data_sources":
                manifest = store.root / "data/snapshots/supply_source.json"
                if manifest.exists():
                    supply = json.loads(manifest.read_text(encoding="utf-8"))
                    row = {"series": "stablecoin_supply", "source": "DefiLlama/totalCirculating.peggedUSD",
                           "unit": supply["unit"], "observations": supply["observations"]}
                    dates = combined.loc[combined.series.eq("stablecoin_cap")].iloc[0]
                    row.update(first_date=dates.first_date, last_date=dates.last_date)
                    combined = pd.concat([combined, pd.DataFrame([row])], ignore_index=True)
            yield name, combined


def pack_results(root=ROOT):
    """Verify every byte before removing loose artifacts; deduplicate identical contents."""
    store = Results(root)
    store.folder.mkdir(parents=True, exist_ok=True)
    loose = [p for p in store.folder.rglob("*") if p.is_file()
             and p.suffix in (".parquet", ".json", ".csv")
             and "tables" not in p.relative_to(store.folder).parts
             and not (p.parent.name == "extensions" and
                      (p.name == "external_panel.parquet" or p.name.endswith("_features.parquet")))]
    names = set(store.index) | {p.relative_to(store.folder).as_posix() for p in loose}
    contents = {name: store.read(name) for name in sorted(names)}
    index = {name: sha256(value).hexdigest() for name, value in contents.items()}
    summaries = list(_summaries(store))
    temporary = store.folder / "research.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(zipfile.ZipInfo("index.json"), json.dumps({"version": 1, "files": index}, sort_keys=True),
                         compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        unique = {index[name]: value for name, value in contents.items()}
        for digest, value in sorted(unique.items()):
            archive.writestr(zipfile.ZipInfo("objects/" + digest), value,
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(temporary) as archive:
        for name, value in contents.items():
            if archive.read("objects/" + index[name]) != value:
                raise ValueError("Result round-trip failed: " + name)
    if store.archive.exists() and temporary.read_bytes() == store.archive.read_bytes():
        temporary.unlink()
    else:
        temporary.replace(store.archive)
    if summaries:
        tables = store.folder / "tables"
        tables.mkdir(exist_ok=True)
        for name, frame in summaries:
            frame.to_csv(tables / (name + ".csv"), index=False, lineterminator="\n")
    for path in loose:
        if not path.resolve().is_relative_to(store.folder.resolve()):
            raise ValueError("Result path escapes folder: " + str(path))
        if path.read_bytes() != contents[path.relative_to(store.folder).as_posix()]:
            raise ValueError("Result changed during packaging: " + str(path))
        path.unlink()
    return {"logical_files": len(index), "unique_contents": len(unique),
            "identical_copies_removed": len(index)-len(unique), "archive_bytes": store.archive.stat().st_size}


@contextmanager
def materialized_results(root=ROOT):
    """Expand only for scientific commands; refuse to overwrite conflicting loose results."""
    store = Results(root)
    for name, digest in store.index.items():
        path = store.folder / name
        if path.exists() and sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Loose result changed: " + name)
    for name in store.index:
        path = store.folder / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(store.read(name))
    yield
    pack_results(root)


def verify_results(root=ROOT):
    store = Results(root)
    for name, expected in store.index.items():
        if sha256(store.read(name)).hexdigest() != expected:
            raise ValueError("Result changed: " + name)
    for name, expected in _summaries(store):
        path = store.folder / "tables" / (name + ".csv")
        if path.read_bytes() != expected.to_csv(index=False, lineterminator="\n").encode("utf-8"):
            raise ValueError("Summary changed: " + name)
