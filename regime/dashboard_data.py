"""Read-only dashboard projection: published history followed by shared live data."""
import os

import pandas as pd
import streamlit as st

from .data import ROOT, config
from .evaluation import backtest, performance
from .external import asof_panel, feature_observations
from .labels import label_predictions
from .live import Service
from .results import Results


@st.cache_resource
def runtime():
    service=Service()
    if os.environ.get("CRYPTO_LIVE_AUTOSTART","1")!="0":
        service.start()
    return service


def version():
    path=ROOT / "results/research.zip"
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def _frame(name,stamp):
    return Results().frame(name)


def read_table(name):
    return _frame(name,version())


@st.cache_data(show_spinner=False)
def _json(name,stamp):
    return Results().json(name)


def read_json(name):
    return _json(name,version())


@st.cache_data(show_spinner=False)
def _summary(name,stamp):
    return Results().table(name)


def summary(name):
    return _summary(name,version())


def append_history(historical,live):
    if live is None or live.empty:
        return historical.copy()
    addition=live.loc[live.index>historical.index.max()]
    return pd.concat([historical,addition]).sort_index()


def market_data(asset):
    return append_history(read_table(asset+"_market.parquet"),runtime().store.frame("market:"+asset))


def model_record(asset,model):
    base=Results().model_path(asset,model).removesuffix(".parquet")
    record=read_json(base+(".json" if base.startswith("extensions/") else ".diagnostics.json"))
    return base,record


def diagnostics(asset,model):
    _,record=model_record(asset,model)
    old=record["diagnostics"] if isinstance(record,dict) else record
    new=runtime().store.json(f"model:{asset}:{model}:diagnostics").get("rows",[])
    by_month={pd.Timestamp(d.get("period",d.get("period_start"))):d for d in [*old,*new]}
    return [by_month[key] for key in sorted(by_month)]


def prediction(asset,model):
    base,record=model_record(asset,model)
    old=record["diagnostics"] if isinstance(record,dict) else record
    historical=label_predictions(asset,model,read_table(base+".parquet"),old)
    live=runtime().store.frame(f"model:{asset}:{model}:predictions")
    if live is not None and "state" in live and "market_regime" not in live:
        live=label_predictions(asset,model,live,diagnostics(asset,model))
    return append_history(historical,live),base


def _stamp(asset):
    names=summary("model_scores").query("asset == @asset").model
    return tuple(runtime().store.json(f"model:{asset}:{m}").get("updated_at",0) for m in names)


@st.cache_data(show_spinner=False,max_entries=16)
def _scores(asset,stamp,archive_version):
    scores=summary("model_scores").query("asset == @asset").copy()
    predictions={name:prediction(asset,name)[0] for name in scores.model}
    returns={row.model:(predictions[row.model]["return_log_score"] if "return_log_score" in predictions[row.model]
                       else predictions[row.model]["log_score"]).dropna()
             for row in scores.itertuples() if pd.notna(row.mean_return_log_score)}
    common=None
    for series in returns.values():
        common=series.index if common is None else common.intersection(series.index)
    joint=predictions["HMM"].log_score.dropna().index.intersection(predictions["GMM"].log_score.dropna().index)
    for index,row in scores.iterrows():
        value=predictions[row.model]
        scores.loc[index,"test_days"]=len(value)
        if "log_score" in value:
            scores.loc[index,"mean_log_score"]=value.log_score.mean()
        if row.model in returns:
            scores.loc[index,"mean_return_log_score"]=returns[row.model].reindex(common).mean()
            scores.loc[index,"return_score_start"]=str(common[0].date())
            scores.loc[index,"return_score_end"]=str(common[-1].date())
            scores.loc[index,"return_score_days"]=len(common)
        if row.model in ("HMM","GMM"):
            scores.loc[index,"mean_log_score"]=value.loc[joint,"log_score"].mean()
            scores.loc[index,"joint_score_start"]=str(joint[0].date())
            scores.loc[index,"joint_score_end"]=str(joint[-1].date())
            scores.loc[index,"joint_score_days"]=len(joint)
        if "state" in value:
            scores.loc[index,"state_changes"]=value.state.diff().fillna(0).ne(0).sum()
        if "noise" in value:
            scores.loc[index,"noise_fraction"]=value.noise.mean()
            scores.loc[index,"mean_membership_strength"]=value.membership_strength.mean()
        scores.loc[index,"nonconverged_refits"]=sum(
            not d.get("converged",d.get("converged_starts",1)>0) for d in diagnostics(asset,row.model))
        scores.loc[index,"last_date"]=str(value.index[-1].date())
    return scores


def current_scores(asset):
    return _scores(asset,_stamp(asset),version())


@st.cache_data(show_spinner=False,max_entries=24)
def _portfolio(asset,model,stamp,archive_version):
    value,base=prediction(asset,model)
    market=market_data(asset)
    end=min(value.index[-1],market.index[-1])
    if end<=pd.Timestamp(config()["test_end"],tz="UTC"):
        rows=summary("portfolio_performance").query("asset == @asset and model == @model").copy()
        paths={(10,strategy):read_table(base+f"_{strategy}_portfolio.parquet")
               for strategy in ["buy_hold","vol_target","regime_aware"]}
        # Other fee scenarios use the same accounting if requested by the dashboard.
    else:
        rows=pd.DataFrame()
        paths={}
    cfg=config()
    dates=pd.date_range(pd.Timestamp(cfg["test_start"],tz="UTC"),end,name="date")
    volatility=(cfg["volatility_target"]/market.rv30).clip(0,1).reindex(dates)
    signals={"buy_hold":pd.Series(1.,index=dates),"vol_target":volatility,
             "regime_aware":volatility*(1-value.p_high_vol.reindex(dates)).fillna(0)}
    calculated=[]
    for cost in cfg["cost_bps"]:
        for strategy,signal in signals.items():
            if (cost,strategy) not in paths:
                paths[cost,strategy]=backtest(market.open.reindex(dates),signal,cost_bps=cost,delay=cfg["execution_delay_days"])
            if rows.empty:
                calculated.append({"asset":asset,"model":model,"strategy":strategy,"cost_bps":cost,
                                   **performance(paths[cost,strategy])})
    return (rows if len(rows) else pd.DataFrame(calculated)),paths


def portfolio_data(asset,model):
    record=runtime().store.json(f"model:{asset}:{model}")
    market=market_data(asset)
    return _portfolio(asset,model,(record.get("updated_at",0),str(market.index[-1])),version())


def panel_data(asset):
    historical=read_table("extensions/external_panel.parquet")
    dates=market_data(asset).index
    observations=runtime().store.frame("external_observations")
    if observations is None or observations.empty:
        return historical.copy()
    panel=asof_panel(pd.concat([observations,feature_observations(observations)]),dates)
    return append_history(historical,panel)


def data_sources():
    rows=summary("data_sources").copy()
    observations=runtime().store.frame("external_observations")
    markets={asset:market_data(asset) for asset in config()["assets"]}
    for index,row in rows.iterrows():
        end=pd.Timestamp(row.last_date)
        end=end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if pd.notna(row.asset):
            frame=markets[row.asset]
            column={"spot":"close","funding":"funding","oi":"oi"}[row.dataset]
            addition=frame.loc[frame.index>end,column].dropna()
            count_column="valid_days"
        elif observations is not None:
            addition=observations.loc[(observations.series==row.series)&(observations.observation_date>end),
                                      ["observation_date","value"]].set_index("observation_date").value
            count_column="observations"
        else:
            continue
        if len(addition):
            rows.loc[index,"last_date"]=str(addition.index.max())
            rows.loc[index,count_column]=row[count_column]+len(addition)
    return rows
