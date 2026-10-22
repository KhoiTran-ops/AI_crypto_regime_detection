"""Live display reads the shared worker cache; viewers never call exchange APIs."""
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .dashboard_views import COLORS, chart, grid_metrics, signal_chart, table
from .live import Service


@st.cache_resource
def runtime():
    service=Service()
    if os.environ.get("CRYPTO_LIVE_AUTOSTART", "1")!="0":
        service.start()
    return service


@st.fragment(run_every="60s")
def live_view(asset, model):
    store=runtime().store
    quotes=store.json("quotes")
    status=store.json(f"model:{asset}:{model}")
    prediction=store.frame(f"model:{asset}:{model}:predictions")
    market=store.frame("market:"+asset)
    fetched=quotes.get("fetched_at")
    price=quotes.get("prices",{}).get(asset)
    current=prediction.iloc[-1] if prediction is not None and len(prediction) else None
    probability=(current.get("p_high_vol",current.get("change_probability")) if current is not None else None)
    risk_label="P(highest-volatility state)"
    if model=="BOCPD":
        risk_label="Change-point probability"
    elif model=="KMeans":
        risk_label="Highest-volatility cluster indicator"
    elif model.startswith("HDBSCAN"):
        risk_label="Membership strength"
        probability=current.get("membership_strength") if current is not None else None
    label=(current.get("market_regime","Change-point signal") if current is not None else "Đang cập nhật")
    volatility=market.rv30.iloc[-1] if market is not None and len(market) else None
    grid_metrics([("Last price · USDT",f"{price:,.2f}" if price is not None else "—"),
                  ("Market regime · "+model,label.split(" · ")[0]),
                  ("30-day annualized volatility",f"{volatility:.1%}" if volatility is not None else "—"),
                  (risk_label,
                   f"{probability:.1%}" if probability is not None else "—")])
    if fetched:
        stamp=pd.Timestamp(fetched,unit="s",tz="UTC")
        st.caption(f"Giá ghi nhận {stamp:%d.%m.%Y %H:%M:%S} UTC · cập nhật mỗi 60 giây")
        if time.time()-fetched>120:
            st.warning("Giá đang lưu từ lần cập nhật trước; kết nối nguồn đang được thử lại.")
    else:
        st.info("Đang kết nối nguồn giá Binance…")
    if current is not None:
        st.caption(f"{label} · kết quả ngày {prediction.index[-1]:%d.%m.%Y} UTC · chỉ dùng nến đã đóng")
    elif status.get("running"):
        st.info("Đang khởi tạo mô hình và đồng bộ các ngày còn thiếu…")
    elif status.get("error"):
        st.info("Mô hình đang đợi dữ liệu: "+status["error"])
    else:
        st.info("Đang đồng bộ dữ liệu ngày và khởi tạo mô hình…")
    worker=store.json("worker:quotes")
    if worker.get("error"):
        st.warning("Nguồn giá: "+worker["error"])
    if market is not None and len(market):
        with st.container(border=True,key="card_live_price"):
            st.subheader("Daily close and market regimes")
            visible=market.tail(180)
            fig=go.Figure(go.Scatter(x=visible.index,y=visible.close,name="Daily close",line=dict(color="#26364b",width=1.5)))
            if prediction is not None and "state" in prediction:
                joined=prediction.join(visible[["close"]],how="inner")
                for name,points in joined.groupby("market_regime"):
                    code=int(points.state.iloc[0])
                    fig.add_trace(go.Scatter(x=points.index,y=points.close,mode="markers",name=name,
                        marker=dict(size=5,color=COLORS[code%len(COLORS)] if code>=0 else COLORS[-1])))
            chart(fig,370)
    if prediction is not None and len(prediction):
        with st.container(border=True,key="card_live_signal"):
            st.subheader("Probabilities and signals · "+model)
            signal_chart(prediction.tail(180),model,asset)
    with st.expander("Trạng thái cập nhật"):
        daily=store.json("daily")
        st.write(daily.get("stage","Đang khởi động"))
        names=store.json("models").get("assets",{}).get(asset,[])
        rows=[]
        for name in names:
            record=store.json(f"model:{asset}:{name}")
            rows.append({"Mô hình":name,"Ngày kết quả UTC":record.get("through","—"),
                         "Trạng thái":"Đang chạy" if record.get("running") else record.get("error") or "Sẵn sàng"})
        if rows:
            table(pd.DataFrame(rows))
        observations=store.frame("external_observations")
        if observations is not None and len(observations):
            coverage=observations.groupby("series").agg(
                observation_date=("observation_date","max"), available_date=("available_date","max"))
            table(coverage.reset_index().rename(columns={"series":"Nguồn dữ liệu", "observation_date":"Ngày quan sát UTC",
                                                       "available_date":"Ngày khả dụng theo quy ước UTC"}))
        for key in ("external_status","derivative_status"):
            for source,error in store.json(key).get("errors",{}).items():
                st.write(source+": "+error)
        if store.json("worker:daily").get("error"):
            st.write(store.json("worker:daily")["error"])
