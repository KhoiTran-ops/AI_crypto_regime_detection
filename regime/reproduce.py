"""Optional full refit in a temporary output directory; leaves the published run intact."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import final
from .data import ROOT, config, normalize_spot, read_archive, save_json


def published_bytes(path):
    from .results import Results
    relative = path.relative_to(ROOT).as_posix()
    return Results().read(relative[len("results/"):]) if relative.startswith("results/") else path.read_bytes()


def verify():
    from .results import verify_results
    verify_results()
    manifest = json.loads(published_bytes(ROOT / "results/artifact_hashes.json"))
    changed = [name for name, expected in manifest.items()
               if hashlib.sha256(published_bytes(ROOT / name)).hexdigest() != expected]
    if changed:
        raise ValueError(f"Published artifacts changed: {changed}")
    lock = json.loads(published_bytes(ROOT / "results/protocol_lock.json"))["protocol"]
    for name, expected in lock["scientific_code"].items():
        if hashlib.sha256((ROOT / name.replace("\\", "/")).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Scientific code changed after protocol lock: {name}")
    current = (ROOT / "requirements.lock").read_bytes()
    if hashlib.sha256(current).hexdigest() != lock["requirements_hash"]:
        baseline = (ROOT / "docs/history/requirements_baseline.lock").read_bytes()
        original_pins = {line for line in baseline.decode().splitlines() if line.strip()}
        current_pins = {line for line in current.decode().splitlines() if line.strip()}
        # Explicit additive extension: none of the original scientific dependencies may change.
        if (hashlib.sha256(baseline).hexdigest() != lock["requirements_hash"]
                or current_pins != original_pins | {"hdbscan==0.8.44"}):
            raise ValueError("Baseline dependency pins changed beyond the declared HDBSCAN extension")
    print(f"Verified {len(manifest)} artifact hashes and all locked scientific source files", flush=True)


def reproduce():
    verify()
    checked = 0
    for path in (ROOT / "data/raw").rglob("*.zip"):
        expected = path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Raw checksum mismatch: {path.name}")
        checked += 1
    for asset in config()["assets"]:
        paths = sorted((ROOT / f"data/raw/data/spot/monthly/klines/{asset}/1d").glob("*.zip"))
        rebuilt = pd.concat([normalize_spot(read_archive(path, header=None), asset) for path in paths]).sort_index()
        saved = pd.read_parquet(ROOT / f"data/processed/{asset}_spot.parquet")
        rebuilt = rebuilt.reindex(saved.index)
        pd.testing.assert_frame_equal(rebuilt, saved, check_freq=False)
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    from . import derivatives
    original_derivative_root = derivatives.ROOT
    with TemporaryDirectory(prefix="data-replay-", dir=ROOT / ".tools") as directory:
        staging = Path(directory)
        raw = staging / "data/raw/funding"
        raw.mkdir(parents=True)
        for asset in config()["assets"]:
            shutil.copy2(ROOT / f"data/raw/funding/{asset}.json", raw / f"{asset}.json")
        try:
            derivatives.ROOT = staging
            derivatives.funding()
            for asset in config()["assets"]:
                pd.testing.assert_frame_equal(
                    pd.read_parquet(staging / f"data/processed/{asset}_funding.parquet"),
                    pd.read_parquet(ROOT / f"data/processed/{asset}_funding.parquet"), check_freq=False)
        finally:
            derivatives.ROOT = original_derivative_root
    for asset in config()["assets"]:
        saved = pd.read_parquet(ROOT / f"data/processed/{asset}_oi.parquet")
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(lambda day: derivatives.oi_day(asset, str(day.date())), saved.index))
        rebuilt = pd.DataFrame(rows).set_index("date").sort_index().reindex(saved.index)
        pd.testing.assert_frame_equal(rebuilt, saved, check_freq=False)
    print("Raw spot, funding and OI replay matched all six processed datasets", flush=True)
    original = final.FINAL
    (ROOT / ".tools").mkdir(exist_ok=True)
    try:
        with TemporaryDirectory(prefix="reproduction-", dir=ROOT / ".tools") as directory:
            final.FINAL = Path(directory)
            final.evaluate()
            names = ["model_scores.csv", "portfolio_performance.csv", "score_comparisons.csv",
                     "portfolio_uncertainty.csv", "method_comparison.csv"]
            for name in names:
                pd.testing.assert_frame_equal(pd.read_csv(original / name), pd.read_csv(final.FINAL / name),
                                              check_exact=False, rtol=1e-8, atol=1e-10)
            maximum = 0.
            for asset in config()["assets"]:
                for model in ("HMM", "GMM", "KMeans", "HMM_return", "MSAR", "HMM_B", "HMM_C", "MS0", "NHHMM"):
                    expected = pd.read_parquet(original / f"{asset}_{model}.parquet")
                    actual = pd.read_parquet(final.FINAL / f"{asset}_{model}.parquet")
                    pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-8, atol=1e-10)
                    maximum = max(maximum, float(np.nanmax(np.abs(
                        expected.select_dtypes("number") - actual.select_dtypes("number")))))
    finally:
        final.FINAL = original
    report = {"status": "passed", "raw_archives_checksum_verified": checked,
        "spot_rebuilt_from_raw": True, "funding_and_oi_rebuilt_from_raw": True, "all_18_models_refitted_without_prediction_cache": True,
        "evaluation_summary_tables_equal": True, "summary_tables_compared": names, "rtol": 1e-8, "atol": 1e-10,
        "maximum_absolute_prediction_difference": maximum,
        "environment": "Same pinned environment; raw cache retained; separate temporary result directory",
        "not_claimed": "Second cold installation, historical data vintages or a second independent data provider"}
    save_json(ROOT / "docs/reproduction.json", report)
    print(json.dumps(report), flush=True)


def seal_extensions():
    """Hash redistributable findings, never third-party source observations."""
    folder = ROOT / "results/extensions"
    from datetime import datetime, timezone
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    record["assembled_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(folder / "run.json", record)
    files = [path for path in folder.iterdir() if path.is_file()
             and path.name not in ("artifact_hashes.json", "external_panel.parquet")
             and not path.name.endswith("_features.parquet")]
    save_json(folder / "artifact_hashes.json", {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)})


def verify_extensions():
    verify()
    folder = ROOT / "results/extensions"
    protocol = json.loads(published_bytes(folder / "protocol.json"))
    checked_inputs = (protocol["hashes"] | protocol.get("supply_addendum", {}).get("hashes", {})
                      | protocol.get("density_addendum", {}).get("hashes", {}))
    for name, expected in checked_inputs.items():
        if hashlib.sha256(published_bytes(ROOT / name)).hexdigest() != expected:
            raise ValueError("Extension input/source changed: " + name)
    artifacts = json.loads(published_bytes(folder / "artifact_hashes.json"))
    for name, expected in artifacts.items():
        if hashlib.sha256(published_bytes(folder / name)).hexdigest() != expected:
            raise ValueError("Extension artifact changed: " + name)
    if json.loads(published_bytes(folder / "run.json"))["fingerprint"] != protocol["fingerprint"]:
        raise ValueError("Run and extension protocol differ")
    print(f"Verified {len(artifacts)} extension artifacts and {len(checked_inputs)} locked inputs", flush=True)
