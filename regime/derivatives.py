"""Funding events and OI archives, retained separately from the spot dataset."""
import hashlib
import json
from urllib.parse import urlencode

import pandas as pd

from .data import ROOT, archive, config, fetch, read_archive, save_json


def funding():
    cfg = config()
    audits = []
    destination = ROOT / "data/processed"
    destination.mkdir(parents=True, exist_ok=True)
    for asset in cfg["assets"]:
        raw_path = ROOT / "data/raw/funding" / f"{asset}.json"
        cutoff = pd.Timestamp(cfg["end"], tz="UTC") + pd.Timedelta(days=1)
        if raw_path.exists():
            events = json.loads(raw_path.read_text())
        else:
            events = []
            cursor = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)
            end = int(cutoff.timestamp() * 1000) - 1
            while cursor < end:
                params = urlencode({"symbol": asset, "startTime": cursor, "endTime": end, "limit": 1000})
                page = json.loads(fetch("https://fapi.binance.com/fapi/v1/fundingRate?" + params))
                if not isinstance(page, list):
                    raise ValueError(f"Funding API error: {str(page)[:200]}")
                if not page:
                    break
                times = [int(item["fundingTime"]) for item in page]
                if min(times) < cursor or max(times) > end or times != sorted(times):
                    raise ValueError("Funding API returned events outside the requested interval")
                events.extend(page)
                cursor = max(times) + 1
            save_json(raw_path, events)
        frame = pd.DataFrame(events)
        frame["time"] = pd.to_datetime(frame.fundingTime, unit="ms", utc=True)
        frame["rate"] = pd.to_numeric(frame.fundingRate, errors="raise")
        conflicts = frame.groupby("time").rate.nunique().gt(1).sum()
        if conflicts:
            raise ValueError(f"Conflicting funding rates for {asset}")
        frame = frame.drop_duplicates("time").sort_values("time")
        frame["date"] = frame.time.dt.floor("D")
        daily = frame.groupby("date").agg(funding=("rate", "sum"), funding_count=("rate", "size"))
        daily = daily.reindex(pd.date_range(daily.index.min(), cutoff - pd.Timedelta(days=1), name="date"))
        # A gap over 12h is conservatively treated as incomplete, not a zero-rate day.
        gaps = frame.time.diff().dt.total_seconds().div(3600)
        bad_days = set(frame.loc[gaps > 12, "date"]) | set(frame.date.shift(1)[gaps > 12].dropna())
        daily.loc[daily.index.isin(bad_days), "funding"] = float("nan")
        daily.to_parquet(destination / f"{asset}_funding.parquet")
        audits.append({"asset": asset, "events": len(frame), "first_event": str(frame.time.min()),
                       "last_event": str(frame.time.max()), "daily_missing": int(daily.funding.isna().sum()),
                       "gaps_over_12h": int((gaps > 12).sum()),
                       "event_count_distribution": daily.funding_count.value_counts().to_dict(),
                       "raw_snapshot": str(raw_path.relative_to(ROOT)),
                       "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                       "source": "https://fapi.binance.com/fapi/v1/fundingRate",
                       "note": "No assumption of exactly three settlements/day; first day can be partial."})
        print(f"Funding {asset}: {len(frame)} events", flush=True)
    save_json(ROOT / "outputs/data_audit/funding.json", audits)
    return audits


def oi_day(asset, day):
    key = f"data/futures/um/daily/metrics/{asset}/{asset}-metrics-{day}.zip"
    item = archive(key)
    frame = read_archive(ROOT / item["file"])
    selected = frame[["create_time", "symbol", "sum_open_interest", "sum_open_interest_value"]].copy()
    selected["time"] = pd.to_datetime(selected.create_time, utc=True)
    selected["oi"] = pd.to_numeric(selected.sum_open_interest, errors="raise")
    selected["oi_quote"] = pd.to_numeric(selected.sum_open_interest_value, errors="raise")
    if selected.symbol.ne(asset).any():
        raise ValueError(f"Unexpected OI symbol: {key}")
    valid = selected[["oi", "oi_quote"]].gt(0).all(axis=1)
    invalid_count = int((~valid).sum())
    selected = selected.loc[valid]
    if selected.empty:
        raise ValueError(f"No valid OI snapshot: {key}")
    conflicts = selected.groupby("time")[["oi", "oi_quote"]].nunique().gt(1).any(axis=1).sum()
    if conflicts:
        raise ValueError(f"Conflicting OI snapshots: {key}")
    duplicates = int(selected.time.duplicated().sum())
    selected = selected.drop_duplicates("time").sort_values("time")
    date = pd.Timestamp(day, tz="UTC")
    if not selected.time.dt.floor("D").eq(date).all():
        raise ValueError(f"OI timestamps outside archive date: {key}")
    last = selected.iloc[-1]
    age = ((date + pd.Timedelta(days=1)) - last.time).total_seconds() / 60
    return {"date": date, "oi": last.oi if age <= 15 else float("nan"),
            "oi_quote": last.oi_quote if age <= 15 else float("nan"),
            "oi_age_minutes": age, "snapshots": len(selected), "duplicates": duplicates,
            "invalid_snapshots": invalid_count}


def open_interest(workers=6):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg, audits = config(), []
    destination = ROOT / "data/processed"
    days = pd.date_range("2021-12-01", cfg["end"])
    for asset in cfg["assets"]:
        summaries, failures = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(oi_day, asset, str(day.date())): day for day in days}
            for number, future in enumerate(as_completed(futures), 1):
                try:
                    summaries.append(future.result())
                except (OSError, ValueError, KeyError) as error:
                    failures.append({"date": str(futures[future].date()), "error": str(error)})
                if number % 200 == 0 or number == len(days):
                    print(f"OI {asset}: {number}/{len(days)}, errors={len(failures)}", flush=True)
        if not summaries:
            raise ValueError(f"No usable OI data for {asset}")
        frame = pd.DataFrame(summaries).set_index("date").sort_index()
        frame = frame.reindex(pd.date_range(days[0], days[-1], tz="UTC", name="date"))
        frame.to_parquet(destination / f"{asset}_oi.parquet")
        audits.append({"asset": asset, "expected_days": len(days), "valid_days": int(frame.oi.notna().sum()),
                       "coverage": float(frame.oi.notna().mean()), "failures": failures,
                       "duplicate_snapshots": int(frame.duplicates.sum()),
                       "invalid_snapshots": int(frame.invalid_snapshots.sum()),
                       "rule": "Last snapshot before midnight; age <=15 minutes; duplicates verified identical."})
        save_json(ROOT / "outputs/data_audit/oi.json", audits)
    return audits
