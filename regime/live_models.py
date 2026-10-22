"""Daily inference adapters over existing estimators and locked model selections."""
import time

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .advanced import bocpd_walk, hdbscan_walk, msgarch_walk
from .benchmarks import cluster_walk, markov_walk
from .data import config
from .extensions import GROUPS
from .labels import label_predictions
from .models import FittedHMM, fit_hmm, walk_forward
from .results import Results
from .supply import COVARIATES

CORE = {"HMM", "GMM", "KMeans", "HMM_return", "MSAR", "MSGARCH", "BOCPD"}


def save_fit(store, key, fitted):
    store.json(key, {"columns":fitted.columns, "order":fitted.order.tolist(), "metadata":fitted.metadata,
        "mean":fitted.scaler.mean_.tolist(), "scale":fitted.scaler.scale_.tolist(),
        "start":fitted.model.startprob_.tolist(), "transition":fitted.model.transmat_.tolist(),
        "means":fitted.model.means_.tolist(), "variance":fitted.variances.tolist()})


def load_fit(store, key):
    value=store.json(key)
    if not value:
        return None
    model=GaussianHMM(len(value["order"]),covariance_type="diag")
    model.n_features=len(value["columns"])
    model.startprob_=np.array(value["start"])
    model.transmat_=np.array(value["transition"])
    model.means_=np.array(value["means"])
    model.covars_=np.array(value["variance"])
    scaler=StandardScaler()
    scaler.mean_,scaler.scale_=np.array(value["mean"]),np.array(value["scale"])
    scaler.n_features_in_=len(value["columns"])
    scaler.feature_names_in_=np.array(value["columns"],dtype=object)
    return FittedHMM(model,scaler,value["columns"],np.array(value["order"]),value["metadata"])


def hmm_update(store, frame, asset, name, k, columns, start, end):
    cfg=config()
    prefix=f"fit:{asset}:{name}:"
    frozen_end=pd.Timestamp(cfg["test_end"],tz="UTC")
    seed_month=frozen_end.replace(day=1)
    reference=load_fit(store,prefix+str(seed_month.date()))
    if reference is None:
        # Reproduce the original state-matching chain once, rather than renaming R states.
        _,_,reference=walk_forward(frame.loc[:frozen_end],columns,k,cfg["test_start"],cfg["test_end"],
            cfg["refit_seeds"],cfg["max_iter"],initial_seeds=cfg["seeds"])
        save_fit(store,prefix+str(seed_month.date()),reference)
    outputs,diagnostics=[],[]
    first=pd.Timestamp(start,tz="UTC").replace(day=1)
    final=pd.Timestamp(end,tz="UTC")
    for month in pd.date_range(seed_month+pd.offsets.MonthBegin(),final,freq="MS"):
        key=prefix+str(month.date())
        fitted=load_fit(store,key)
        train=frame.loc[frame.index<month].dropna(subset=columns)
        if fitted is None:
            fitted=fit_hmm(train,columns,k,cfg["refit_seeds"],cfg["max_iter"],reference)
            save_fit(store,key,fitted)
        reference=fitted
        if month<first:
            continue
        evaluation=frame.loc[(frame.index>=month)&(frame.index<=final)&(frame.index<month+pd.offsets.MonthBegin())].dropna(subset=columns)
        if evaluation.empty:
            continue
        _,previous=fitted.filter(train)
        if evaluation.index[0]-train.index[-1]!=pd.Timedelta(days=1):
            previous=None
        prediction,_=fitted.filter(evaluation,previous=previous)
        prediction["fit_cutoff"]=train.index[-1]
        prediction["stress_state"]=fitted.stress_state
        outputs.append(prediction)
        diagnostics.append({"period_start":str(month),"transition":fitted.transition.tolist(),
                            "stress_state":fitted.stress_state,**fitted.metadata})
    if not outputs:
        raise ValueError("No completed observations available for this model")
    return pd.concat(outputs),diagnostics


def estimate(store, frame, asset, name, start, end):
    cfg=config()
    archive=Results()
    protocol=archive.json("protocol_lock.json")["protocol"]
    core=cfg["core_features"]
    if name.startswith("HMM"):
        columns=core.copy()
        k=protocol["core"][asset]["k"]
        if name=="HMM_return":
            columns=["return"]
            k=protocol["benchmarks"][asset][name]
        elif name in ("HMM_B","HMM_C"):
            cohort=name[-1]
            k=protocol["cohorts"][asset][cohort]
            columns += ["funding"]+(["oi_change"] if cohort=="C" else [])
            frame=frame.loc["2021-12-02" if cohort=="C" else "2019-11-28":]
        return hmm_update(store,frame,asset,name,k,columns,start,end)
    if name in ("GMM","KMeans"):
        k=protocol["benchmarks"][asset]["GMM"] if name=="GMM" else protocol["core"][asset]["k"]
        return cluster_walk(frame,core,name,k,start,end)
    if name=="MSAR":
        prediction,diagnostics,_=markov_walk(frame,name,protocol["benchmarks"][asset]["MSAR"],start,end)
        return prediction,diagnostics
    if name in ("MS0","NHHMM"):
        frame=frame.loc["2019-11-28":].copy()
        frame["funding_lag2"]=frame.funding.shift(2)
        frame["volume_lag2"]=frame.volume_z.shift(2)
        covariates=["funding_lag2","volume_lag2"]
        frame=frame.dropna(subset=["return",*covariates])
        prediction,diagnostics,_=markov_walk(frame,name,2,start,end,covariates if name=="NHHMM" else None)
        return prediction,diagnostics
    record=archive.json(archive.model_path(asset,name).removesuffix(".parquet")+".json")
    decision=record["decision"]
    if name=="MSGARCH":
        return msgarch_walk(frame,start,end)
    if name=="BOCPD":
        return bocpd_walk(frame,cfg["test_start"],end,decision["hazard"],decision["prior_weight"]),[]
    if name.startswith("HDBSCAN"):
        return hdbscan_walk(frame,core,start,end,int(decision["min_cluster_size"]))
    group=name.split("_",1)[1]
    cols=([c+"_delay7_lag2" for c in GROUPS["MACRO"]] if group=="MACRO_DELAY7" else
          [c+"_lag2" for c in (COVARIATES if group=="SUPPLY" else GROUPS[group])])
    frame=frame.dropna(subset=["return",*cols])
    if frame.empty or frame.index[-1]<pd.Timestamp(end,tz="UTC"):
        raise ValueError("Waiting for input series: "+", ".join(cols))
    prediction,diagnostics,_=markov_walk(frame,"NHHMM" if name.startswith("TVTP_") else "MS0",2,start,end,
                                        cols if name.startswith("TVTP_") else None)
    return prediction,diagnostics


def update_models(store, frames, cutoff, core_only):
    scores=Results().table("model_scores")
    by_asset={asset:scores.loc[scores.asset.eq(asset),"model"].drop_duplicates().tolist() for asset in frames}
    store.json("models", {"assets":by_asset})
    frozen=pd.Timestamp(config()["test_end"],tz="UTC")
    for asset,frame in frames.items():
        for name in by_asset[asset]:
            is_core=name in CORE or name.startswith("HDBSCAN")
            if is_core != core_only:
                continue
            key=f"model:{asset}:{name}"
            status=store.json(key)
            if status.get("through")==str(cutoff.date()) and not status.get("error"):
                continue
            prior=store.frame(key+":predictions")
            start=(prior.index[-1]+pd.Timedelta(days=1) if prior is not None and len(prior) else frozen+pd.Timedelta(days=1)).replace(day=1)
            store.json(key,{**status,"running":True,"error":""})
            try:
                with threadpool_limits(limits=1):
                    prediction,diagnostics=estimate(store,frame,asset,name,str(start.date()),str(cutoff.date()))
                prediction=prediction.loc[prediction.index>frozen]
                if prediction.empty or prediction.index[-1]!=cutoff:
                    raise ValueError("Waiting for complete input data through "+str(cutoff.date()))
                if not (pd.to_datetime(prediction.fit_cutoff,utc=True)<prediction.index).all():
                    raise ValueError("Non-causal live model training cutoff")
                pcols=[c for c in prediction if c.startswith("p_") and c[2:].isdigit()]
                if pcols:
                    values=prediction[pcols]
                    if not np.isfinite(values).all().all() or (values<0).any().any() or not np.allclose(values.sum(axis=1),1):
                        raise ValueError("Invalid live state probabilities")
                    prediction[pcols]=values.div(values.sum(axis=1),axis=0)
                    if name!="MSGARCH":
                        prediction["p_high_vol"]=prediction[pcols[-1]] if not name.startswith("HMM") else prediction.p_high_vol.clip(0,1)
                    prediction=label_predictions(asset,name,prediction,diagnostics)
                elif "state" in prediction:
                    prediction=label_predictions(asset,name,prediction,diagnostics)
                if prior is not None:
                    prediction=pd.concat([prior,prediction.loc[prediction.index>prior.index[-1]]])
                store.frame(key+":predictions",prediction)
                old_diagnostics=store.json(key+":diagnostics").get("rows",[])
                by_month={d.get("period",d.get("period_start")):d for d in old_diagnostics+diagnostics}
                store.json(key+":diagnostics", {"rows":list(by_month.values())})
                store.json(key,{"through":str(prediction.index[-1].date()),"updated_at":time.time(),"running":False,"error":""})
            except Exception as error:
                store.json(key,{**status,"running":False,"error":str(error)[:500],"updated_at":time.time()})
