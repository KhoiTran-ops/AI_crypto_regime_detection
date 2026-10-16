"""Binance archives: checksum verification, preserved raw files and explicit audit."""
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPOT_COLUMNS = ["open_time", "open", "high", "low", "close", "base_volume", "close_time",
                "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "BinanceAcademicResearch/0.1"}),
                         timeout=30) as response:
                body = response.read(50_000_001)
                if len(body) > 50_000_000:
                    raise ValueError("Unexpected response larger than 50 MB")
                return body
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            time.sleep(min(float(error.headers.get("Retry-After", 2 ** attempt)), 15))
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def archive(key):
    target = ROOT / "data" / "raw" / key
    target.parent.mkdir(parents=True, exist_ok=True)
    url = "https://data.binance.vision/" + key
    checksum_path = target.with_suffix(target.suffix + ".CHECKSUM")
    if not checksum_path.exists():
        checksum_path.write_bytes(fetch(url + ".CHECKSUM"))
    expected = checksum_path.read_text().split()[0].lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid checksum response: {key}")
    if not target.exists():
        body = fetch(url)
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {key}")
        target.write_bytes(body)
    body = target.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError(f"Cached archive changed: {key}")
    return {"file": str(target.relative_to(ROOT)).replace("\\", "/"), "url": url,
            "sha256": expected, "bytes": len(body),
            "verified_at_utc": datetime.now(timezone.utc).isoformat()}


def read_archive(path, **kwargs):
    with zipfile.ZipFile(path) as zipped:
        members = [item for item in zipped.infolist() if item.filename.endswith(".csv")]
        if len(members) != 1 or members[0].file_size > 50_000_000:
            raise ValueError(f"Unexpected ZIP content: {path}")
        return pd.read_csv(io.BytesIO(zipped.read(members[0])), **kwargs)


def normalize_spot(frame, symbol):
    if frame.shape[1] != 12:
        raise ValueError(f"Unexpected spot schema: {frame.shape}")
    frame = frame.copy()
    frame.columns = SPOT_COLUMNS
    numeric = frame.apply(pd.to_numeric, errors="raise")
    stamp = numeric.open_time.astype("int64")
    # Binance spot archives changed milliseconds to microseconds in January 2025.
    micros = stamp.where(stamp >= 100_000_000_000_000, stamp * 1000)
    numeric.index = pd.to_datetime(micros.to_numpy(), unit="us", utc=True)
    numeric.index.name = "date"
    invalid = ((numeric.low > numeric[["open", "close"]].min(axis=1))
               | (numeric.high < numeric[["open", "close"]].max(axis=1))
               | (numeric.low <= 0) | (numeric.high < numeric.low)
               | (numeric.base_volume < 0) | (numeric.quote_volume < 0))
    if invalid.any():
        raise ValueError(f"OHLC or volume invalid for {symbol}: {invalid.sum()} rows")
    numeric["symbol"] = symbol
    return numeric.drop(columns=["open_time", "close_time", "ignore"])


def download_spot(workers=4):
    cfg = config()
    jobs = [f"data/spot/monthly/klines/{asset}/1d/{asset}-1d-{month}.zip"
            for asset in cfg["assets"]
            for month in pd.period_range(cfg["raw_start"], cfg["end"], freq="M")]
    return download_jobs(jobs, "spot", workers)


def download_jobs(jobs, kind, workers=4):
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(archive, job): job for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                records.append(future.result())
            except (OSError, ValueError) as error:
                failures.append({"key": futures[future], "error": str(error)})
            if number % 50 == 0 or number == len(jobs):
                print(f"{kind}: {number}/{len(jobs)}, errors={len(failures)}", flush=True)
    save_json(ROOT / "data" / f"manifest_{kind}.json",
              {"records": sorted(records, key=lambda item: item["file"]), "failures": failures})
    return records, failures


def build_spot():
    cfg, audits = config(), []
    expected = pd.date_range(cfg["raw_start"], cfg["end"], tz="UTC", name="date")
    destination = ROOT / "data" / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    for asset in cfg["assets"]:
        paths = sorted((ROOT / "data/raw/data/spot/monthly/klines" / asset / "1d").glob("*.zip"))
        if not paths:
            raise ValueError(f"No archives for {asset}")
        frame = pd.concat([normalize_spot(read_archive(path, header=None), asset) for path in paths])
        duplicates = int(frame.index.duplicated().sum())
        if duplicates:
            groups = frame.reset_index().groupby("date")
            if (groups.nunique(dropna=False).max(axis=1) > 1).any():
                raise ValueError(f"Conflicting duplicate dates for {asset}")
            frame = frame.loc[~frame.index.duplicated()]
        frame = frame.sort_index().reindex(expected)
        missing = frame.close.isna()
        audits.append({"asset": asset, "expected_days": len(expected), "valid_days": int((~missing).sum()),
                       "coverage": float((~missing).mean()), "duplicate_rows": duplicates,
                       "first": str(frame.first_valid_index()), "last": str(frame.last_valid_index()),
                       "missing_dates": [str(day.date()) for day in expected[missing]],
                       "partial_first_day": "2017-08-17; first minute 04:00 UTC",
                       "nonpositive_volume_days": int((frame.quote_volume <= 0).sum())})
        frame.to_parquet(destination / f"{asset}_spot.parquet")
    save_json(ROOT / "outputs/data_audit/spot.json", audits)
    print(json.dumps(audits, ensure_ascii=False), flush=True)
    if any(item["coverage"] < .995 for item in audits):
        raise ValueError("Core coverage below predeclared 99.5% gate")
    return audits
