"""Incremental daily inputs; original observations and research artifacts stay immutable."""
import hashlib
import io
import json
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .data import config, normalize_spot
from .extensions import GROUPS
from .external import _fred, asof_panel, feature_observations, validate_observations
from .features import build_features
from .results import Results
from .supply import COVARIATES

DAY = pd.Timedelta(days=1)


def closed_bars(rows, asset, boundary):
    boundary = pd.Timestamp(boundary)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    opened = pd.to_datetime(pd.to_numeric(frame[0]), unit="ms", utc=True)
    closed = pd.to_datetime(pd.to_numeric(frame[6]), unit="ms", utc=True)
    eligible = (opened < boundary.floor("D")) & (closed < boundary)
    frame = frame.loc[eligible].copy()
    if frame.empty:
        return pd.DataFrame()
    opened, closed = opened[eligible], closed[eligible]
    if not opened.eq(opened.dt.floor("D")).all() or not (closed-opened).eq(DAY-pd.Timedelta(milliseconds=1)).all():
        raise ValueError("Unexpected daily UTC bar boundaries")
    result = normalize_spot(frame, asset)
    if not np.isfinite(result.select_dtypes(include="number")).all().all():
        raise ValueError("Non-finite daily bar")
    return result


def merge_days(old, addition):
    if addition.empty:
        return old
    addition = addition.sort_index()
    overlap = old.index.intersection(addition.index)
    common = old.columns.intersection(addition.columns)
    if len(overlap) and not old.loc[overlap, common].eq(addition.loc[overlap, common]).all().all():
        raise ValueError("Conflicting existing daily observations")
    result = pd.concat([old, addition.loc[~addition.index.isin(old.index)]]).sort_index()
    if result.index.has_duplicates or not result.index.to_series().diff().dropna().eq(DAY).all():
        raise ValueError("Daily data gap; inference paused")
    return result


def update_spot(store, client, cutoff):
    cfg = config()
    if time.time()-store.json("exchange").get("fetched_at", 0) >= 86400:
        info = client.json("https://api.binance.com/api/v3/exchangeInfo?"+urlencode({"symbols": json.dumps(cfg["assets"],separators=(",", ":"))}))
        minute = next(r["limit"] for r in info["rateLimits"] if r["rateLimitType"] == "REQUEST_WEIGHT"
                      and r["interval"] == "MINUTE" and r["intervalNum"] == 1)
        store.json("exchange", {"minute_weight": minute, "fetched_at": time.time()})
    frames = {}
    for asset in cfg["assets"]:
        prior = store.frame("market:"+asset)
        old = prior if prior is not None else Results().frame(asset+"_market.parquet")
        start = old.index[-1]+DAY
        while start <= cutoff:
            end = min(cutoff, start+999*DAY)
            params = {"symbol": asset, "interval": "1d", "startTime": int(start.timestamp()*1000),
                      "endTime": int((end+DAY).timestamp()*1000)-1, "limit": 1000}
            rows = client.json("https://api.binance.com/api/v3/klines?"+urlencode(params))
            extra = closed_bars(rows, asset, cutoff+DAY)
            if extra.empty or extra.index[0] != start or extra.index[-1] != end:
                raise ValueError(f"Incomplete spot coverage: {asset}, {start.date()}–{end.date()}")
            old = merge_days(old, extra)
            start = old.index[-1]+DAY
        generated = build_features(old)
        new = generated.loc[generated.index > pd.Timestamp(cfg["end"], tz="UTC")]
        market = pd.concat([old.loc[:cfg["end"]], new])
        store.frame("market:"+asset, market)
        store.json("source:spot:"+asset, {"through": str(market.index[-1].date()), "fetched_at": time.time()})
        frames[asset] = market
    return frames


def cached_body(store, client, name, url):
    key = "source:"+name
    prior = store.json(key)
    body = store.blob("raw:"+name)
    if body is None or time.time()-prior.get("fetched_at", 0) >= 3600:
        body = client.get(url)
        store.blob("raw:"+name, body)
        store.json(key, {"fetched_at": time.time(), "sha256": hashlib.sha256(body).hexdigest(), "url": url})
    return body



def fred_daily(store, client, cutoff):
    """One durable FRED download batch per UTC rollover, including failed attempts."""
    day=str((cutoff+DAY).date())
    key="fred_daily"
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row=db.execute("SELECT value FROM cache WHERE name=?",(key,)).fetchone()
        record=json.loads(row[0]) if row else {}
        claimed=record.get("day","")<day
        if claimed:
            record.update(day=day,attempted_at=time.time(),error="FRED download in progress")
            db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",(key,json.dumps(record).encode()))
    if claimed:
        try:
            start=pd.Timestamp(config()["end"],tz="UTC")-60*DAY
            tables=[]
            for series in ["VIXCLS","DGS10","DGS2","SP500"]:
                url="https://fred.stlouisfed.org/graph/fredgraph.csv?"+urlencode(
                    {"id":series,"cosd":str(start.date()),"coed":str(cutoff.date())})
                body=client.get(url)
                table=_fred(body)[["observation_date",series]]
                dates=pd.to_datetime(table.observation_date,utc=True,errors="raise")
                values=pd.to_numeric(table[series],errors="raise")
                if table.empty or dates.duplicated().any() or not dates.between(start,cutoff).all():
                    raise ValueError("Invalid FRED date range: "+series)
                if not np.isfinite(values.dropna()).all():
                    raise ValueError("Invalid FRED values: "+series)
                tables.append(table.assign(**{series:values}))
                store.blob("raw:fred-"+series,body)
                store.json("source:fred-"+series,{"fetched_at":time.time(),"url":url,
                                                "sha256":hashlib.sha256(body).hexdigest()})
            table=tables[0]
            for other in tables[1:]:
                table=table.merge(other,on="observation_date",how="outer",validate="one_to_one")
            body=table.to_csv(index=False).encode()
            record.update(error="",last_success_at=time.time(),last_success_cutoff=str(cutoff.date()),
                          sha256=hashlib.sha256(body).hexdigest())
            with store.connection() as db:
                db.executemany("INSERT OR REPLACE INTO cache VALUES (?,?)",
                               [("raw:fred-daily",body),(key,json.dumps(record).encode())])
        except Exception as error:
            record.update(error=str(error) or type(error).__name__,finished_at=time.time())
            store.json(key,record)
    body=store.blob("raw:fred-daily")
    if body is None:
        raise RuntimeError(record.get("error") or "FRED has no cached observations")
    return _fred(body)


def funding(store, client, asset, cutoff):
    prior = store.frame("funding:"+asset)
    start = prior.index[-1]+DAY if prior is not None and len(prior) else pd.Timestamp(config()["end"], tz="UTC")+DAY
    if start > cutoff:
        return prior
    cursor, events = int(start.timestamp()*1000), []
    end = int((cutoff+DAY).timestamp()*1000)-1
    while cursor <= end:
        params = {"symbol": asset, "startTime": cursor, "endTime": end, "limit": 1000}
        page = client.json("https://fapi.binance.com/fapi/v1/fundingRate?"+urlencode(params))
        if not page:
            break
        stamps = [int(row["fundingTime"]) for row in page]
        if min(stamps) < cursor or max(stamps) > end or stamps != sorted(set(stamps)):
            raise ValueError("Invalid funding event pagination")
        events.extend(page)
        cursor = stamps[-1]+1
        if len(page) < 1000:
            break
    if not events:
        raise ValueError("Funding not yet available")
    raw = pd.DataFrame(events)
    raw["time"] = pd.to_datetime(raw.fundingTime, unit="ms", utc=True)
    raw["funding"] = pd.to_numeric(raw.fundingRate)
    daily = raw.groupby(raw.time.dt.floor("D")).agg(funding=("funding", "sum"), funding_count=("funding", "size"))
    daily.index.name = "date"
    gaps = raw.time.diff().gt(pd.Timedelta(hours=12))
    bad = set(raw.loc[gaps, "time"].dt.floor("D")) | set(raw.time.shift()[gaps].dropna().dt.floor("D"))
    daily.loc[daily.index.isin(bad), "funding"] = np.nan
    daily = daily.reindex(pd.date_range(start, cutoff, tz="UTC", name="date"))
    result = pd.concat([prior, daily]) if prior is not None else daily
    store.frame("funding:"+asset, result)
    return result


def interest(store, client, asset, cutoff):
    import zipfile
    prior = store.frame("oi:"+asset)
    start = prior.index[-1]+DAY if prior is not None and len(prior) else pd.Timestamp(config()["end"], tz="UTC")+DAY
    rows = []
    for day in pd.date_range(start, cutoff, tz="UTC"):
        key = f"data/futures/um/daily/metrics/{asset}/{asset}-metrics-{day.date()}.zip"
        url = "https://data.binance.vision/"+key
        try:
            digest = client.get(url+".CHECKSUM").decode().split()[0]
            body = client.get(url)
        except Exception:
            if rows:
                value = pd.DataFrame(rows).set_index("date")
                store.frame("oi:"+asset, pd.concat([prior, value]) if prior is not None else value)
            raise
        if hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("OI archive checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].file_size > 50_000_000:
                raise ValueError("Unexpected OI archive size/schema")
            table = pd.read_csv(archive.open(members[0]))
        stamp = pd.to_datetime(table.create_time, utc=True)
        valid = table.sum_open_interest.gt(0) & table.sum_open_interest_value.gt(0)
        table = table.loc[valid].assign(time=stamp[valid]).sort_values("time")
        if table.empty or not table.symbol.eq(asset).all() or not table.time.dt.floor("D").eq(day).all():
            raise ValueError("Invalid OI archive")
        if table.groupby("time")[["sum_open_interest", "sum_open_interest_value"]].nunique().gt(1).any().any():
            raise ValueError("Conflicting OI snapshots")
        last = table.iloc[-1]
        age = (day+DAY-last.time).total_seconds()/60
        rows.append({"date": day, "oi": last.sum_open_interest if age <= 15 else np.nan,
                     "oi_quote": last.sum_open_interest_value if age <= 15 else np.nan})
    addition = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    result = pd.concat([prior, addition]) if prior is not None else addition
    if len(result):
        store.frame("oi:"+asset, result)
    return result


def external_panel(store, client, cutoff, dates):
    start = pd.Timestamp(config()["end"], tz="UTC")-60*DAY
    sources = {
        "fred": None,
        "dxy": "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?"+urlencode({"period1":int(start.timestamp()), "period2":int((cutoff+DAY).timestamp()), "interval":"1d"}),
        "caps": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"+urlencode({"assets":"btc,eth", "metrics":"CapMrktCurUSD", "start_time":str(start.date()), "end_time":str(cutoff.date()), "page_size":10000}),
        "stablecoins": "https://stablecoins.llama.fi/stablecoincharts/all",
        "tvl": "https://api.llama.fi/v2/historicalChainTvl"}
    rows = []
    business = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    def append(name, stamps, values, unit, source, us=False):
        for date, value in zip(stamps, values):
            if value is None or pd.isna(value):
                continue
            date = pd.Timestamp(date)
            date = (date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")).normalize()
            if not start <= date <= cutoff:
                continue
            rows.append({"series":name, "observation_date":date, "available_date":date+(business if us else DAY),
                         "value":float(value), "unit":unit, "source":source})
    errors = {}
    for name, url in sources.items():
        try:
            if name == "fred":
                table=fred_daily(store,client,cutoff)
                if store.json("fred_daily").get("error"):
                    errors["fred"]=store.json("fred_daily")["error"]
            else:
                body=cached_body(store,client,name,url)
            if name == "fred":
                for col, label, unit in [("VIXCLS","vix","index"),("DGS10","ust10","percent"),("DGS2","ust2","percent"),("SP500","sp500","index")]:
                    append(label,table.observation_date,table[col],unit,"FRED",True)
            elif name == "dxy":
                data = json.loads(body)["chart"]["result"][0]
                if data["meta"]["symbol"] != "DX-Y.NYB" or data["meta"]["instrumentType"] != "INDEX":
                    raise ValueError("Unexpected DXY instrument")
                append("dxy",pd.to_datetime(data["timestamp"],unit="s",utc=True),data["indicators"]["quote"][0]["close"],"index","Yahoo/ICE-DXY",True)
            elif name == "caps":
                data = json.loads(body)
                if data.get("next_page_url"):
                    raise ValueError("Incomplete CoinMetrics pagination")
                for asset in ("btc","eth"):
                    group=[r for r in data["data"] if r["asset"]==asset]
                    append(asset+"_market_cap",[r["time"] for r in group],[r["CapMrktCurUSD"] for r in group],"USD","CoinMetrics")
            elif name == "stablecoins":
                data=json.loads(body)
                stamps=pd.to_datetime([int(r["date"]) for r in data],unit="s",utc=True)
                for field, label in [("totalCirculatingUSD","stablecoin_cap"),("totalCirculating","stablecoin_supply")]:
                    append(label,stamps,[r.get(field,{}).get("peggedUSD") for r in data],"USD","DefiLlama")
            else:
                data=json.loads(body)
                append("defi_tvl",pd.to_datetime([r["date"] for r in data],unit="s",utc=True),[r["tvl"] for r in data],"USD","DefiLlama")
        except Exception as error:
            errors[name]=str(error)[:250]
    store.json("external_status", {"errors":errors, "fetched_at":time.time()})
    if not rows:
        return pd.DataFrame(index=dates)
    observations=validate_observations(pd.DataFrame(rows))
    store.frame("external_observations",observations)
    return asof_panel(pd.concat([observations,feature_observations(observations)]), dates)


def update_controls(store, client, frames, cutoff):
    cfg=config()
    errors={}
    for asset, frame in frames.items():
        for name, operation in (("funding",funding),("oi",interest)):
            try:
                values=operation(store,client,asset,cutoff)
            except Exception as error:
                errors[asset+":"+name]=str(error)[:250]
                values=store.frame(name+":"+asset)
            if values is not None and len(values):
                for column in values:
                    frame.loc[values.index.intersection(frame.index),column]=values[column]
        frame["oi_change"]=np.log(frame.oi/frame.oi.shift())
        store.frame("market:"+asset,frame)
    dates=frames[cfg["assets"][0]].index
    panel=external_panel(store,client,cutoff,dates)
    correlation=frames[cfg["assets"][0]]["return"].rolling(30).corr(frames[cfg["assets"][1]]["return"])
    for asset, frame in frames.items():
        full=frame.join(panel)
        full["btc_eth_corr30"]=correlation
        full["amihud"]=np.expm1(full["return"]).abs()/full.quote_volume.where(full.quote_volume>0)
        full["log_illiquidity30"]=np.log(full.amihud.rolling(30).mean().where(lambda v:v>0))
        full["market_cap_growth"]=full.get(asset[:3].lower()+"_market_cap_growth",np.nan)
        for columns in [*GROUPS.values(),COVARIATES]:
            for column in columns:
                full[column+"_lag2"]=full.get(column,pd.Series(np.nan,index=dates)).shift(cfg["execution_delay_days"])
        for column in GROUPS["MACRO"]:
            full[column+"_delay7_lag2"]=full[column+"_lag2"].shift(7)
        base=Results().frame("extensions/"+asset+"_features.parquet")
        full=pd.concat([base,full.loc[full.index>base.index[-1]]])
        store.frame("features:"+asset,full)
        frames[asset]=full
    store.json("derivative_status", {"errors":errors,"fetched_at":time.time()})
    return frames
