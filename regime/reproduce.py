"""Optional full refit in a temporary output directory; leaves the published run intact."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import final
from .data import ROOT, config, normalize_spot, read_archive, save_json


def verify():
    manifest = json.loads((ROOT / "results/artifact_hashes.json").read_text(encoding="utf-8"))
    changed = [name for name, expected in manifest.items()
               if not (ROOT / name).exists() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    if changed:
        raise ValueError(f"Published artifacts changed: {changed}")
    lock = json.loads((ROOT / "results/protocol_lock.json").read_text(encoding="utf-8"))["protocol"]
    for name, expected in lock["scientific_code"].items():
        if hashlib.sha256((ROOT / name.replace("\\", "/")).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Scientific code changed after protocol lock: {name}")
    if hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest() != lock["requirements_hash"]:
        raise ValueError("Locked dependency file changed")
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
