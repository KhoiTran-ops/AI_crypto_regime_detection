"""Five research views over a shared result store."""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import softmax

from .data import ROOT, config
from .results import Results, materialized_results

COLORS = ["#147d78", "#4263eb", "#c98a25", "#b85651", "#875db2", "#6b7280"]
STRATEGIES = {"buy_hold": "Mua và giữ", "vol_target": "Theo biến động", "regime_aware": "Theo trạng thái"}
SERIES = {"close": "Giá đóng cửa · USDT", "quote_volume": "Khối lượng · USDT", "rv30": "Biến động 30 ngày · năm hóa",
          "funding": "Funding · tổng ngày", "oi": "Open interest · đơn vị tài sản", "vix": "VIX", "dxy": "DXY",
          "ust10": "Treasury 10 năm · %", "ust2": "Treasury 2 năm · %", "sp500": "S&P 500 · chỉ số giá",
          "stablecoin_cap": "Vốn hóa stablecoin · USD", "stablecoin_supply": "Cung stablecoin · USD mệnh giá",
          "defi_tvl": "DeFi TVL · USD", "btc_market_cap": "Vốn hóa BTC · USD", "eth_market_cap": "Vốn hóa ETH · USD"}


def version():
    path = ROOT / "results/research.zip"
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def _frame(name, stamp):
    return Results().frame(name)


def read_table(name):
    return _frame(name, version())


@st.cache_data(show_spinner=False)
def _json(name, stamp):
    return Results().json(name)


def read_json(name):
    return _json(name, version())


@st.cache_data(show_spinner=False)
def _summary(name, stamp):
    return Results().table(name)


def summary(name):
    return _summary(name, version())


def chart(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=25, b=10),
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#526175"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      colorway=COLORS, legend=dict(orientation="h", y=1.13, x=0),
                      xaxis=dict(gridcolor="#f0f2f5"), yaxis=dict(gridcolor="#edf0f4"))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def grid_metrics(items):
    for column, (label, value) in zip(st.columns(len(items)), items, strict=True):
        with column, st.container(border=True, key="card_" + label):
            st.metric(label, value)


def table(frame, formats=None):
    labels = {"asset": "Tài sản", "model": "Mô hình", "first_date": "Từ ngày", "last_date": "Đến ngày",
              "test_days": "Ngày đánh giá", "cohort": "Nhóm biến", "split": "Tập mẫu", "start": "Từ ngày",
              "end": "Đến ngày", "complete_feature_days": "Quan sát hợp lệ", "strategy": "Chiến lược",
              "volatility": "Biến động năm hóa", "max_drawdown": "Drawdown lớn nhất", "turnover": "Vòng quay",
              "ending_value": "Giá trị cuối kỳ", "block_days": "Độ dài block", "bootstrap_repetitions": "Lần bootstrap",
              "n_common": "Quan sát chung", "mean_difference": "Chênh lệch trung bình", "ci_low": "Cận dưới 95%",
              "ci_high": "Cận trên 95%", "noise_fraction": "Tỷ lệ noise", "state_changes": "Lần đổi trạng thái"}
    counts = ["Số quan sát", "test_days", "n_common", "complete_feature_days", "k", "block_days",
              "bootstrap_repetitions", "state_changes", "nonconverged_refits", "n_parameters"]
    formatting = {column: "{:,.0f}" for column in counts if column in frame}
    formatting.update(formats or {})
    formatting = {labels.get(column, column): value for column, value in formatting.items()}
    st.dataframe(frame.rename(columns=labels).style.format(formatting, na_rep="—"),
                 hide_index=True, width="stretch")


def prediction(asset, model):
    path = Results().model_path(asset, model)
    return read_table(path), path.removesuffix(".parquet")


def probabilities(value):
    return [c for c in value if re.fullmatch(r"p_\d+", c)]


def signal_chart(value, model):
    cols = probabilities(value)
    if cols:
        fig = go.Figure()
        for i, col in enumerate(cols):
            fig.add_trace(go.Scatter(x=value.index, y=value[col], name=f"R{i+1}", stackgroup="one",
                                    line=dict(width=.5, color=COLORS[i % len(COLORS)])))
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text={"KMeans": "Chỉ báo gán cụm", "GMM": "Xác suất thành phần"}.get(model, "Xác suất lọc"))
    elif "change_probability" in value:
        fig = px.line(value, x=value.index, y="change_probability",
                      labels={"change_probability": "Xác suất điểm thay đổi", "date": "Ngày UTC"})
    else:
        plotted = value.copy()
        plotted["Cụm"] = plotted.state.map(lambda x: "Noise" if x == -1 else f"R{int(x)+1}")
        fig = px.scatter(plotted, x=plotted.index, y="membership_strength" if "membership_strength" in value else "state",
                         color="Cụm", color_discrete_sequence=COLORS, labels={"date": "Ngày UTC"})
    chart(fig, 260)


def overview(asset, model):
    market = read_table(f"{asset}_market.parquet")
    value, _ = prediction(asset, model)
    last, current = value.iloc[-1], market.loc[value.index[-1]]
    state = f"R{int(last.state)+1}" if "state" in value and last.state >= 0 else "—"
    if "noise" in value and last.noise:
        state = "Noise"
    grid_metrics([("Giá đóng cửa · USDT", f"{current.close:,.2f}"), ("Trạng thái · " + model, state),
                  ("Biến động 30 ngày · năm hóa", f"{current.rv30:.1%}"),
                  ("Chỉ báo cụm σ cao nhất" if model == "KMeans" else "P(trạng thái σ cao nhất)",
                   f"{last.p_high_vol:.1%}" if "p_high_vol" in value else "—")])
    main, side = st.columns([2.25, 1])
    with main, st.container(border=True, key="card_price"):
        st.subheader("Giá và trạng thái thị trường")
        selected = st.date_input("Khoảng hiển thị", (value.index[0].date(), value.index[-1].date()),
                                 min_value=value.index[0].date(), max_value=value.index[-1].date())
        visible = value.loc[str(selected[0]):str(selected[-1])] if len(selected) == 2 else value
        joined = visible.join(market[["close"]])
        fig = go.Figure(go.Scatter(x=joined.index, y=joined.close, name="Giá", line=dict(color="#26364b", width=1.4)))
        if "state" in joined:
            for i, state_id in enumerate(sorted(joined.state.unique())):
                points = joined.loc[joined.state.eq(state_id)]
                fig.add_trace(go.Scatter(x=points.index, y=points.close, mode="markers",
                    name="Noise" if state_id == -1 else f"R{int(state_id)+1}",
                    marker=dict(size=4, color=COLORS[i % len(COLORS)])))
        fig.update_yaxes(title_text="USDT")
        chart(fig, 365)
    with side, st.container(border=True, key="card_study"):
        st.subheader("Hồ sơ nghiên cứu")
        st.metric("Tài sản", asset.replace("USDT", " / USDT"))
        st.metric("Cấu hình mô hình / tài sản", int(summary("model_scores").asset.eq(asset).sum()))
        st.metric("Quan sát đánh giá", f"{len(value):,} ngày")
        st.caption(f"{value.index[0]:%d.%m.%Y} — {value.index[-1]:%d.%m.%Y}")
        st.caption("Đơn vị thời gian: ngày UTC")
    with st.container(border=True, key="card_signal"):
        st.subheader("Phân bố trạng thái" if probabilities(value) else "Diễn biến tín hiệu")
        signal_chart(visible, model)
    left, right = st.columns(2)
    for column, title, field in ((left, "BTC / ETH · giá chuẩn hóa", "close"),
                                  (right, "BTC / ETH · tương quan 30 ngày", "correlation")):
        with column, st.container(border=True, key="card_compare_" + field):
            st.subheader(title)
            frames = {a: read_table(f"{a}_market.parquet") for a in ("BTCUSDT", "ETHUSDT")}
            if field == "close":
                fig = go.Figure()
                for a, f in frames.items():
                    prices = f.loc[value.index, "close"]
                    fig.add_trace(go.Scatter(x=prices.index, y=100*prices/prices.iloc[0], name=a[:3]))
            else:
                corr = frames["BTCUSDT"]["return"].rolling(30).corr(frames["ETHUSDT"]["return"]).loc[value.index]
                fig = go.Figure(go.Scatter(x=corr.index, y=corr, name="Tương quan", line_color=COLORS[0]))
                fig.update_yaxes(range=[-1, 1])
            chart(fig, 250)


def models(asset, model):
    scores = summary("model_scores").query("asset == @asset").copy()
    value, _ = prediction(asset, model)
    selected = scores.loc[scores.model.eq(model)].iloc[0]
    grid_metrics([("Cấu hình", str(len(scores))), ("Quan sát", str(len(value))),
                  ("Log score · return", f"{selected.mean_return_log_score:.4f}" if pd.notna(selected.mean_return_log_score) else "—"),
                  ("Số lần fit chưa hội tụ", str(int(selected.nonconverged_refits)))])
    with st.container(border=True, key="card_model_scores"):
        st.subheader("Đánh giá mô hình")
        target = st.selectbox("Chỉ tiêu", ["Mật độ return", "Mật độ đa biến · nhóm lõi", "Phân cụm"])
        if target == "Phân cụm":
            selected_models = scores[scores.model.str.startswith(("HDBSCAN", "KMeans"))]
            table(selected_models.reindex(columns=["model", "test_days", "state_changes", "noise_fraction", "mean_membership_strength"]),
                  {"noise_fraction": "{:.1%}", "mean_membership_strength": "{:.3f}"})
        else:
            metric = "mean_return_log_score" if target == "Mật độ return" else "mean_log_score"
            comparable = scores.dropna(subset=[metric])
            if target != "Mật độ return":
                comparable = comparable.loc[comparable.score_group.eq("joint_core")]
            comparable = comparable.sort_values(metric)
            fig = px.bar(comparable, x=metric, y="model", orientation="h", color="model",
                         color_discrete_sequence=COLORS, text=metric,
                         labels={metric: "Log density trung bình", "model": ""})
            fig.update_traces(texttemplate="%{x:.4f}", textposition="outside")
            fig.update_layout(showlegend=False, hovermode="closest")
            chart(fig, max(270, len(comparable)*28+65))
        st.caption("Log score càng cao càng tốt. Mật độ đa biến nhóm lõi: return, biến động và khối lượng tương đối.")
    with st.container(border=True, key="card_score_comparisons"):
        st.subheader("Chênh lệch điểm và khoảng tin cậy 95%")
        comparison = summary("score_comparisons").query("asset == @asset").copy()
        comparison["left"] = comparison.left.fillna(comparison.model)
        comparison["right"] = comparison.right.fillna(comparison.reference)
        comparison["Chỉ tiêu"] = np.where(comparison.left.eq("HMM") & comparison.right.eq("GMM"),
                                           "Joint · nhóm lõi", "Return")
        table(comparison[["Chỉ tiêu", "left", "right", "n_common", "mean_difference", "ci_low", "ci_high"]],
              {"mean_difference": "{:.5f}", "ci_low": "{:.5f}", "ci_high": "{:.5f}"})
    with st.expander("Lựa chọn số trạng thái"):
        selection = read_table("model_selection.csv").query("asset == @asset")
        table(selection[["k", "validation_log_score", "train_bic", "n_parameters", "min_posterior_days"]],
              {"validation_log_score": "{:.4f}", "train_bic": "{:.1f}", "min_posterior_days": "{:.1f}"})
    with st.expander("Bảng điểm toàn bộ mô hình"):
        table(scores, {"mean_return_log_score": "{:.5f}", "mean_log_score": "{:.5f}"})
        st.download_button("Tải bảng điểm", scores.to_csv(index=False), file_name=f"{asset}_models.csv")
    with st.container(border=True, key="card_model_signal"):
        st.subheader("Tín hiệu · " + model)
        signal_chart(value, model)


def portfolio(asset, model):
    performance = summary("portfolio_performance").query("asset == @asset and model == @model")
    cost = st.segmented_control("Chi phí giao dịch · bps", [5, 10, 25], default=10)
    cost = 10 if cost is None else cost
    view = performance.loc[performance.cost_bps.eq(cost)].copy()
    active = view.loc[view.strategy.eq("regime_aware")].iloc[0]
    grid_metrics([("CAGR · theo trạng thái", f"{active.cagr:.1%}"), ("Sharpe", f"{active.sharpe:.2f}"),
                  ("Drawdown lớn nhất", f"{active.max_drawdown:.1%}"), ("Tỷ trọng trung bình", f"{active.average_exposure:.1%}")])
    _, base = prediction(asset, model)
    equity, drawdown = go.Figure(), go.Figure()
    for strategy, label in STRATEGIES.items():
        path = read_table(base + f"_{strategy}_portfolio.parquet")
        equity.add_trace(go.Scatter(x=path.index, y=path.equity, name=label))
        drawdown.add_trace(go.Scatter(x=path.index, y=path.drawdown, name=label))
    for title, fig in (("Giá trị danh mục · 10 bps", equity), ("Drawdown · 10 bps", drawdown)):
        with st.container(border=True, key="card_portfolio_" + title):
            st.subheader(title)
            if fig is drawdown:
                fig.update_yaxes(tickformat=".0%")
            chart(fig, 310 if fig is equity else 245)
    with st.container(border=True, key="card_portfolio_table"):
        st.subheader(f"Hiệu quả và rủi ro · {cost} bps")
        view["strategy"] = view.strategy.map(STRATEGIES)
        table(view[["strategy", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "turnover", "ending_value"]],
              {"cagr": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.3f}", "sortino": "{:.3f}",
               "max_drawdown": "{:.2%}", "turnover": "{:.2f}", "ending_value": "{:.3f}"})
        intervals = summary("portfolio_uncertainty").query("asset == @asset and model == @model")
        table(intervals[["sharpe_difference_ci_low", "sharpe_difference_ci_high", "block_days", "bootstrap_repetitions"]])
        st.download_button("Tải hiệu quả danh mục", performance.to_csv(index=False), file_name=f"{asset}_{model}_portfolio.csv")


def states(asset, model):
    value, base = prediction(asset, model)
    store = Results()
    path = base + (".json" if base.startswith("extensions/") else ".diagnostics.json")
    record = read_json(path)
    diagnostics = record["diagnostics"] if isinstance(record, dict) else record
    with st.container(border=True, key="card_state_signal"):
        st.subheader("Xác suất và tín hiệu · " + model)
        signal_chart(value, model)
    transitions = [d for d in diagnostics if "transition" in d]
    left, right = st.columns(2)
    if transitions:
        latest = transitions[-1]
        matrix = np.asarray(latest["transition"])
        labels = [f"R{i+1}" for i in range(len(matrix))]
        if model == "MSGARCH":
            labels = [f"S{i+1} thô" for i in range(len(matrix))]
        with left, st.container(border=True, key="card_transition"):
            st.subheader("Ma trận chuyển trạng thái")
            chart(px.imshow(matrix, x=labels, y=labels, text_auto=".2f", zmin=0, zmax=1,
                            color_continuous_scale=["#f1f7f6", "#147d78"],
                            labels={"x": "Ngày kế tiếp", "y": "Ngày hiện tại", "color": "Xác suất"}), 325)
    with right if transitions else st.container(), st.container(border=True, key="card_regime_profile"):
        st.subheader("Đặc trưng trạng thái")
        if model == "HMM":
            table(summary("regime_profiles").query("asset == @asset").drop(columns="asset"),
                  {"mean_return_daily": "{:.3%}", "volatility_annual": "{:.1%}", "occupancy": "{:.1%}",
                   "persistence": "{:.3f}", "expected_duration_days": "{:.1f}"})
            st.caption("Đặc trưng của lần fit đầu kỳ đánh giá, từ dữ liệu đến 31.12.2024.")
        elif "state" in value:
            counts = value.state.value_counts(normalize=True).rename_axis("state").reset_index(name="Tỷ lệ quan sát")
            counts["state"] = counts.state.map(lambda v: "Noise" if v == -1 else f"R{int(v)+1}")
            table(counts, {"Tỷ lệ quan sát": "{:.1%}"})
        elif model == "BOCPD":
            table(pd.DataFrame([record["decision"]]))
    coefficient_path = base + ("_coefficients.csv" if base.startswith("extensions/") else ".coefficients.csv")
    if store.exists(coefficient_path):
        with st.container(border=True, key="card_coefficients"):
            st.subheader("Hệ số mô hình")
            coefficients = read_table(coefficient_path)
            if isinstance(record, dict):
                covariates = ["Intercept", *(record["decision"].get("covariates") or [])]
                coefficients["covariate"] = [covariates[int(p.split(".tvtp")[1])] if ".tvtp" in p else p
                                             for p in coefficients.parameter]
            if diagnostics and "raw_high_vol_state" in diagnostics[0]:
                mapping = pd.DataFrame(diagnostics)[["period", "raw_high_vol_state"]]
                coefficients = coefficients.merge(mapping, on="period", validate="many_to_one")
            table(coefficients)
    if model == "MSGARCH":
        rows = []
        for fit in diagnostics:
            theta = np.asarray(fit["parameters"])
            weights = softmax(np.column_stack([theta[2:4], theta[4:6], np.zeros(2)]), axis=1)
            variance = np.exp(theta[:2])*fit["scale"]**2
            for state_id in range(2):
                rows.append({"Tháng": fit["period"][:10], "Trạng thái thô": f"S{state_id+1}",
                    "alpha": weights[state_id, 0], "beta": weights[state_id, 1],
                    "omega": variance[state_id]*weights[state_id, 2], "Sigma năm hóa": np.sqrt(365*variance[state_id])})
        with st.container(border=True, key="card_garch_parameters"):
            st.subheader("Tham số GARCH")
            table(pd.DataFrame(rows))


def data(asset, model):
    market = read_table(f"{asset}_market.parquet")
    sources = summary("data_sources")
    grid_metrics([("Bắt đầu dữ liệu spot", "17.08.2017"), ("Chốt dữ liệu", "30.09.2026"),
                  ("Tài sản", "BTC / ETH"), ("Chuỗi vĩ mô / on-chain", str(sources.series.notna().sum()))])
    with st.container(border=True, key="card_data_explorer"):
        st.subheader("Khám phá dữ liệu")
        series = st.selectbox("Chỉ số", list(SERIES), format_func=SERIES.get)
        if series in market:
            chart(px.line(market, x=market.index, y=series, labels={series: SERIES[series], "date": "Ngày UTC"}), 360)
        else:
            panel_path = ROOT / "results/extensions/external_panel.parquet"
            if panel_path.exists():
                panel = pd.read_parquet(panel_path)
                chart(px.line(panel, x=panel.index, y=series, labels={series: SERIES[series], "date": "Ngày khả dụng · UTC"}), 360)
            elif st.button("Tải chuỗi dữ liệu", icon=":material/download:"):
                from .external import build_external
                from .supply import add_supply
                with st.spinner("Đang tải và chuẩn hóa dữ liệu…"), materialized_results():
                    add_supply(build_external())
                st.rerun()
    with st.container(border=True, key="card_data_sources"):
        st.subheader("Nguồn và độ phủ dữ liệu")
        source_view = sources.copy()
        source_view["Chỉ số"] = source_view.series.fillna(source_view.dataset).map(lambda v: SERIES.get(v, {"spot": "Spot OHLCV"}.get(v, v)))
        source_view["Nguồn"] = source_view.source.fillna("Binance")
        source_view["Số quan sát"] = source_view.observations.fillna(source_view.valid_days)
        for col in ("first_date", "last_date"):
            source_view[col] = pd.to_datetime(source_view[col], utc=True, format="mixed").dt.strftime("%d.%m.%Y")
        table(source_view[["Chỉ số", "asset", "Nguồn", "first_date", "last_date", "Số quan sát"]])
    with st.container(border=True, key="card_sample_counts"):
        st.subheader("Cỡ mẫu theo nhóm biến")
        counts = summary("sample_counts").query("asset == @asset")
        table(counts[["cohort", "split", "start", "end", "complete_feature_days"]])
    with st.container(border=True, key="card_methods"):
        st.subheader("Phương pháp nghiên cứu")
        cfg = config()
        st.markdown(f"Dữ liệu ngày UTC; lợi suất log, biến động lăn, khối lượng tương đối, thanh khoản, "
                    f"funding/OI và các biến vĩ mô/on-chain. Train đến **{cfg['train_end']}**; "
                    f"validation **{cfg['validation_start']} – {cfg['validation_end']}**; "
                    f"đánh giá **{cfg['test_start']} – {cfg['test_end']}**.")
        st.markdown("Mô hình gồm HMM, GMM, K-means, Markov switching, TVTP, MS-GARCH, BOCPD và HDBSCAN. "
                    "Các mô hình tái ước lượng dùng cửa sổ mở rộng theo tháng; BOCPD cập nhật tuần tự. "
                    "Mật độ dự báo được tính trước khi quan sát ngày cần dự báo.")
        st.markdown("Danh mục mua/giữ, điều chỉnh biến động và điều chỉnh theo trạng thái; mục tiêu biến động 40%/năm, "
                    "tỷ trọng 0–100%, lãi tiền mặt 0. Tín hiệu cuối ngày t giao dịch ở open t+2. "
                    "Chi phí 5/10/25 bps; bootstrap block 30 ngày, 1.000 lần để tính khoảng tin cậy 95%.")


PAGES = {"Tổng quan": (":material/space_dashboard:", overview), "Mô hình": (":material/analytics:", models),
         "Danh mục": (":material/account_balance_wallet:", portfolio), "Trạng thái": (":material/hub:", states),
         "Dữ liệu": (":material/database:", data)}
