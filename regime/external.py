"""Supplemental controls only; frozen raw bundle and point-in-time assumptions are explicit."""
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .data import ROOT, config, fetch, save_json

SNAPSHOT = ROOT / "data/snapshots"
DEST = ROOT / "results/extensions"
SOURCES = {
    "fred": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS,DGS10,DGS2,SP500&cosd=2017-08-17&coed=2026-09-30",
    "dxy": "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?period1=1502928000&period2=1790812800&interval=1d",
    "capitalization": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc,eth&metrics=CapMrktCurUSD&start_time=2017-08-17&end_time=2026-09-30&page_size=10000",
    "stablecoins": "https://stablecoins.llama.fi/stablecoincharts/all",
    "tvl": "https://api.llama.fi/v2/historicalChainTvl",
}


def validate_observations(frame):
    if frame.duplicated(["series", "observation_date"]).any():
        raise ValueError("Duplicate source observation; resolve conflicts explicitly")
    if (frame.available_date < frame.observation_date).any():
        raise ValueError("Observation used before its date")
    if not np.isfinite(frame.value).all():
        raise ValueError("Non-finite source observation")
    return frame.sort_values(["series", "observation_date"]).reset_index(drop=True)


def asof_panel(observations, dates):
    panel = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date").as_unit("ns"))
    for series, group in observations.groupby("series", sort=True):
        group = group.sort_values(["available_date", "observation_date"]).copy()
        group["available_date"] = group.available_date.dt.as_unit("ns")
        group["observation_date"] = group.observation_date.dt.as_unit("ns")
        known = pd.merge_asof(pd.DataFrame({"date": panel.index}),
                              group[["available_date", "observation_date", "value"]],
                              left_on="date", right_on="available_date", direction="backward")
        panel[series] = known.value.to_numpy()
        panel[f"{series}_age_days"] = ((known.date - known.observation_date).dt.total_seconds() / 86400).to_numpy()
    return panel


def _fred(body):
    if body[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            tables = [pd.read_csv(io.BytesIO(archive.read(name)), na_values=["."])
                      for name in archive.namelist() if name.endswith(".csv")]
        frame = tables[0]
        for other in tables[1:]:
            frame = frame.merge(other, on="observation_date", how="outer", validate="one_to_one")
        return frame
    return pd.read_csv(io.BytesIO(body), na_values=["."])


def normalize(payloads):
    rows = []
    lower, upper = (pd.Timestamp(config()[key], tz="UTC") for key in ("raw_start", "end"))
    business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    def append(series, dates, values, unit, source, business=False):
        for date, value in zip(dates, values):
            if pd.isna(value):
                continue
            date = pd.Timestamp(date).normalize()
            date = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
            if not lower <= date <= upper:
                continue
            available = date + (business_day if business else pd.Timedelta(days=1))
            rows.append({"series": series, "observation_date": date, "available_date": available,
                         "value": float(value), "unit": unit, "source": source})
    fred = _fred(payloads["fred"])
    for name, series, unit in [("VIXCLS", "vix", "index"), ("DGS10", "ust10", "percent"),
                               ("DGS2", "ust2", "percent"), ("SP500", "sp500", "index")]:
        append(series, fred.observation_date, fred[name], unit, "FRED/" + name, business=True)
    dxy = json.loads(payloads["dxy"])["chart"]["result"][0]
    if dxy["meta"]["symbol"] != "DX-Y.NYB" or dxy["meta"]["instrumentType"] != "INDEX":
        raise ValueError("Expected ICE DXY INDEX, not a currency future")
    append("dxy", pd.to_datetime(dxy["timestamp"], unit="s", utc=True),
           dxy["indicators"]["quote"][0]["close"], "index", "Yahoo/ICE-DXY", business=True)
    caps = json.loads(payloads["capitalization"])
    if caps.get("next_page_url"):
        raise ValueError("Coin Metrics pagination incomplete")
    for asset in ("btc", "eth"):
        data = [row for row in caps["data"] if row["asset"] == asset]
        append(asset + "_market_cap", [row["time"] for row in data],
               [row["CapMrktCurUSD"] for row in data], "USD", "CoinMetrics/CapMrktCurUSD")
    stable = json.loads(payloads["stablecoins"])
    append("stablecoin_cap", pd.to_datetime([int(row["date"]) for row in stable], unit="s", utc=True),
           [row.get("totalCirculatingUSD", {}).get("peggedUSD", np.nan) for row in stable],
           "USD", "DefiLlama/USD-pegged")
    tvl = json.loads(payloads["tvl"])
    append("defi_tvl", pd.to_datetime([row["date"] for row in tvl], unit="s", utc=True),
           [row["tvl"] for row in tvl], "USD", "DefiLlama/TVL")
    return validate_observations(pd.DataFrame(rows))


def feature_observations(observations):
    outputs = []
    for series, group in observations.groupby("series"):
        group = group.sort_values("observation_date").copy()
        values = group.value.where(group.value > 0)
        gap = group.observation_date.diff().dt.days
        # Session returns use consecutive reported closes; aggregate changes require consecutive UTC days.
        transforms = {}
        if series in ("dxy", "sp500"):
            transforms[series + "_last_session_return"] = np.log(values / values.shift())
        elif series == "vix":
            transforms["log_vix"] = np.log(values)
        elif series.startswith("ust"):
            transforms[series + "_change"] = group.value.diff() / 100
        else:
            transforms[series + "_growth"] = np.log(values / values.shift()).where(gap.eq(1))
        for name, transformed in transforms.items():
            output = group.assign(series=name, value=transformed, unit="log-change or decimal")
            outputs.append(output.dropna(subset=["value"]))
    return pd.concat(outputs, ignore_index=True)


def canonical_hash(observations):
    return hashlib.sha256(observations.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%SZ",
        float_format="%.17g", lineterminator="\n").encode()).hexdigest()


def build_external(refresh=False):
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    bundle = SNAPSHOT / "external_raw.zip"
    receipt = ROOT / "outputs/external_download.json"
    expected_path = SNAPSHOT / "external_sources.json"
    if bundle.exists() and not refresh:
        manifest = json.loads(receipt.read_text(encoding="utf-8"))
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
            raise ValueError("Frozen external bundle checksum mismatch")
        with zipfile.ZipFile(bundle) as archive:
            payloads = {name: archive.read(name) for name in SOURCES}
    else:
        if bundle.exists():
            raise ValueError("Preserve this snapshot; refresh in a separate checkout after removing only its external bundle")
        payloads = {name: fetch(url) for name, url in SOURCES.items()}
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, body in payloads.items():
                archive.writestr(name, body)
        manifest = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "cutoff": config()["end"], "sources": {
                        name: {"url": SOURCES[name], "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
                        for name, body in payloads.items()},
                    "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    "availability": "Assumed next US federal business day for US market closes; next UTC day for aggregates. Not certified historical vintages."}
        save_json(receipt, manifest)
    for name, body in payloads.items():
        if hashlib.sha256(body).hexdigest() != manifest["sources"][name]["sha256"]:
            raise ValueError("External source checksum mismatch: " + name)
    observations = normalize(payloads)
    normalized_hash = canonical_hash(observations)
    if expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        if normalized_hash != expected["normalized_observations_sha256"]:
            raise ValueError("Provider history differs from published normalized snapshot; preserve this checkout and create a new data version")
    else:
        save_json(expected_path, {"normalized_observations_sha256": normalized_hash,
            "cutoff": config()["end"], "sources": SOURCES,
            "redistribution": "External observations/raw are downloaded locally, not redistributed in Git. FRED SP500 requires prior permission to reproduce."})
    observations.to_parquet(SNAPSHOT / "external_observations.parquet", index=False)
    derived = feature_observations(observations)
    dates = pd.date_range(config()["analysis_start"], config()["end"], tz="UTC", name="date")
    panel = asof_panel(pd.concat([observations, derived]), dates)
    panel["ust_term_spread"] = (panel.ust10 - panel.ust2) / 100
    panel["btc_share_btc_eth"] = panel.btc_market_cap / (panel.btc_market_cap + panel.eth_market_cap)
    panel.to_parquet(DEST / "external_panel.parquet")
    audit = observations.groupby("series").agg(first_date=("observation_date", "min"),
        last_date=("observation_date", "max"), observations=("value", "size"), minimum=("value", "min"),
        unit=("unit", "first"), source=("source", "first"))
    audit["nonpositive"] = observations.assign(bad=observations.value.le(0)).groupby("series").bad.sum()
    audit.to_csv(DEST / "source_coverage.csv")
    print(audit.to_string(), flush=True)
    return panel
