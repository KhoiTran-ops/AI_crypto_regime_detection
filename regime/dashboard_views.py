"""Five research views over a shared result store."""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import softmax

from .dashboard_data import (
    current_scores,
    data_sources,
    market_data,
    model_record,
    panel_data,
    portfolio_data,
    prediction,
    read_table,
    runtime,
    summary,
)
from .dashboard_data import (
    diagnostics as model_diagnostics,
)
from .data import ROOT, config
from .labels import label_note, state_label
from .results import Results

COLORS = ["#0d9488", "#2563eb", "#d68a12", "#dc5266", "#8754c8", "#64748b"]
STRATEGIES = {"buy_hold": "Buy-and-hold", "vol_target": "Volatility targeting", "regime_aware": "Regime-aware"}
SERIES = {"close": "Close price · USDT", "quote_volume": "Quote volume · USDT", "rv30": "30-day annualized volatility",
          "funding": "Funding · tổng ngày", "oi": "Open interest · đơn vị tài sản", "vix": "VIX", "dxy": "DXY",
          "ust10": "10-year Treasury yield · %", "ust2": "2-year Treasury yield · %", "sp500": "S&P 500 · price index",
          "stablecoin_cap": "Stablecoin market cap · USD", "stablecoin_supply": "Stablecoin nominal supply · USD",
          "defi_tvl": "DeFi TVL · USD", "btc_market_cap": "BTC market cap · USD", "eth_market_cap": "ETH market cap · USD"}


def chart(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=25, b=10),
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#526175"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      colorway=COLORS, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      xaxis=dict(gridcolor="#f0f2f5"), yaxis=dict(gridcolor="#edf0f4"))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def grid_metrics(items):
    for i, (column, (label, value)) in enumerate(zip(st.columns(len(items)), items, strict=True)):
        with column, st.container(border=True, key=f"card_metric_{i}"):
            st.metric(label, value)


def table(frame, formats=None):
    labels = {"asset": "Tài sản", "model": "Mô hình", "first_date": "Từ ngày", "last_date": "Đến ngày",
              "test_days": "Ngày đánh giá", "cohort": "Nhóm biến", "split": "Tập mẫu", "start": "Từ ngày",
              "end": "Đến ngày", "complete_feature_days": "Quan sát hợp lệ", "strategy": "Chiến lược",
              "volatility": "Annualized volatility", "max_drawdown": "Maximum drawdown", "turnover": "Turnover",
              "ending_value": "Terminal portfolio value", "block_days": "Block length (days)", "bootstrap_repetitions": "Bootstrap replications",
              "n_common": "Quan sát chung", "mean_difference": "Mean difference", "ci_low": "95% CI lower",
              "ci_high": "95% CI upper", "noise_fraction": "Noise fraction", "state_changes": "State changes",
              "regime": "Mã", "mean_return_daily": "Mean daily log return", "volatility_annual": "Annualized volatility",
              "occupancy": "Posterior-weighted occupancy", "persistence": "Persistence", "expected_duration_days": "Expected duration (days)"}
    counts = ["Số quan sát", "test_days", "n_common", "complete_feature_days", "k", "block_days",
              "bootstrap_repetitions", "state_changes", "nonconverged_refits", "n_parameters"]
    formatting = {column: "{:,.0f}" for column in counts if column in frame}
    formatting.update(formats or {})
    formatting = {labels.get(column, column): value for column, value in formatting.items()}
    st.dataframe(frame.rename(columns=labels).style.format(formatting, na_rep="—"),
                 hide_index=True, width="stretch")


def probabilities(value):
    return [c for c in value if re.fullmatch(r"p_\d+", c)]


def signal_chart(value, model, asset):
    cols = probabilities(value)
    if cols:
        fig = go.Figure()
        if model == "MSGARCH":
            high = value.display_high_state.eq(1)
            signals = [("Low-volatility regime", np.where(high, value.p_0, value.p_1)),
                       ("High-volatility regime", np.where(high, value.p_1, value.p_0))]
        else:
            signals = [(state_label(asset, model, i, k=len(cols)), value[col]) for i, col in enumerate(cols)]
        for i, (name, values) in enumerate(signals):
            fig.add_trace(go.Scatter(x=value.index, y=values, name=name, stackgroup="one",
                                    line=dict(width=.5, color=COLORS[i % len(COLORS)])))
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text={"KMeans": "Cluster assignment indicator", "GMM": "Component probability"}.get(model, "Filtered probability"))
    elif "change_probability" in value:
        fig = px.line(value, x=value.index, y="change_probability",
                      labels={"change_probability": "Change-point probability", "date": "Ngày UTC"})
    else:
        plotted = value.copy()
        plotted["Cluster"] = plotted.market_regime
        fig = px.scatter(plotted, x=plotted.index, y="membership_strength" if "membership_strength" in value else "state",
                         color="Cluster", color_discrete_sequence=COLORS, labels={"date": "Ngày UTC"})
    chart(fig, 305 if cols else 260)


def overview(asset, model):
    market = market_data(asset)
    value, _ = prediction(asset, model)
    last, current = value.iloc[-1], market.iloc[-1]
    quotes = runtime().store.json("quotes")
    price = quotes.get("prices", {}).get(asset)
    state = last.market_regime.split(" · ", 1)[0] if "market_regime" in value else "—"
    risk_label = "Highest-volatility cluster indicator" if model == "KMeans" else "P(highest-volatility state)"
    risk_value = f"{last.p_high_vol:.1%}" if "p_high_vol" in value else "—"
    if "membership_strength" in value:
        risk_label, risk_value = "Membership strength", f"{last.membership_strength:.3f}"
    elif model == "BOCPD":
        risk_label, risk_value = "MAP run length · days", f"{int(last.map_run_length):,}"
    grid_metrics([("Last price · USDT" if price is not None else "Close price · USDT", f"{price if price is not None else current.close:,.2f}"),
                  ("Change-point probability" if model == "BOCPD" else "Market regime · " + model,
                   f"{last.change_probability:.1%}" if model == "BOCPD" else state),
                  ("30-day annualized volatility", f"{current.rv30:.1%}"), (risk_label, risk_value)])
    if quotes.get("fetched_at"):
        stamp = pd.Timestamp(quotes["fetched_at"], unit="s", tz="UTC")
        st.caption(f"Giá ghi nhận {stamp:%d.%m.%Y %H:%M:%S} UTC · mô hình đến {value.index[-1]:%d.%m.%Y} UTC")
    if "market_regime" in value:
        st.caption("Trạng thái: " + last.market_regime)
    main, side = st.columns([2.25, 1])
    with main, st.container(border=True, key="card_price"):
        st.subheader("Price and market regimes")
        selection_key, end_key = "display_range_v2_" + asset, "display_end_v2_" + asset
        start = max(market.index[0].date(), pd.Timestamp(config()["test_start"]).date())
        default = st.session_state.get(selection_key) or (start, market.index[-1].date())
        if len(default) == 2 and default[-1] == st.session_state.get(end_key):
            default = (default[0], market.index[-1].date())
        selected = st.date_input("Khoảng hiển thị", default, min_value=market.index[0].date(),
                                 max_value=market.index[-1].date(), key=f"display_dates_v2_{asset}_{market.index[-1].date()}")
        st.session_state[selection_key] = selected
        st.session_state[end_key] = market.index[-1].date()
        window = slice(str(selected[0]), str(selected[-1])) if selected and len(selected) == 2 else slice(None)
        prices = market
        visible = value.loc[window]
        fig = go.Figure(go.Scatter(x=prices.index, y=prices.close, name="Daily close", line=dict(color="#26364b", width=1.4)))
        if "state" in value:
            joined = value.join(prices[["close"]], how="inner")
            for name, points in joined.groupby("market_regime", sort=True):
                state_id = int(points.state.iloc[0])
                fig.add_trace(go.Scatter(x=points.index, y=points.close, mode="markers", name=name,
                    marker=dict(size=4, color=COLORS[state_id % len(COLORS)] if state_id >= 0 else COLORS[-1])))
        if selected and len(selected) == 2:
            fig.update_xaxes(range=[str(selected[0]), str(selected[-1])])
        fig.update_layout(dragmode="pan")
        fig.update_yaxes(title_text="USDT")
        chart(fig, 405 if model == "HMM" else 365)
        st.caption(label_note(model))
    with side, st.container(border=True, key="card_study"):
        st.subheader("Hồ sơ nghiên cứu")
        st.metric("Tài sản", asset.replace("USDT", " / USDT"))
        st.metric("Cấu hình mô hình / tài sản", int(summary("model_scores").asset.eq(asset).sum()))
        st.metric("Quan sát mô hình", f"{len(value):,} ngày")
        st.caption(f"{value.index[0]:%d.%m.%Y} — {value.index[-1]:%d.%m.%Y}")
        st.caption("Đơn vị thời gian: ngày UTC")
    with st.container(border=True, key="card_signal"):
        st.subheader("Regime probabilities" if probabilities(value) else "Diễn biến tín hiệu")
        if len(visible):
            signal_chart(visible, model, asset)
        else:
            st.info("Chưa có kết quả mô hình trong khoảng ngày đã chọn.")
    left, right = st.columns(2)
    for column, title, field in ((left, "BTC / ETH · normalized prices", "close"),
                                  (right, "BTC / ETH · 30-day correlation", "correlation")):
        with column, st.container(border=True, key="card_compare_" + field):
            st.subheader(title)
            frames = {a: market_data(a) for a in ("BTCUSDT", "ETHUSDT")}
            if field == "close":
                fig = go.Figure()
                for a, frame in frames.items():
                    prices = frame.loc[window, "close"]
                    if len(prices):
                        fig.add_trace(go.Scatter(x=prices.index, y=100*prices/prices.iloc[0], name=a[:3]))
            else:
                corr = frames["BTCUSDT"]["return"].rolling(30).corr(frames["ETHUSDT"]["return"]).loc[window]
                fig = go.Figure(go.Scatter(x=corr.index, y=corr, name="Correlation", line_color=COLORS[0]))
                fig.update_yaxes(range=[-1, 1])
            chart(fig, 250)


def models(asset, model):
    scores = current_scores(asset)
    value, _ = prediction(asset, model)
    selected = scores.loc[scores.model.eq(model)].iloc[0]
    grid_metrics([("Cấu hình", str(len(scores))), ("Quan sát", str(len(value))),
                  ("Log score · return", f"{selected.mean_return_log_score:.4f}" if pd.notna(selected.mean_return_log_score) else "—"),
                  ("Non-converged refits", str(int(selected.nonconverged_refits)))])
    st.caption(f"{value.index[0]:%d.%m.%Y} — {value.index[-1]:%d.%m.%Y} UTC")
    with st.container(border=True, key="card_model_scores"):
        st.subheader("Đánh giá mô hình")
        target = st.selectbox("Chỉ tiêu", ["Return density", "Joint density · core features", "Clustering"])
        if target == "Clustering":
            selected_models = scores[scores.model.str.startswith(("HDBSCAN", "KMeans"))]
            table(selected_models.reindex(columns=["model", "test_days", "state_changes", "noise_fraction", "mean_membership_strength"]),
                  {"noise_fraction": "{:.1%}", "mean_membership_strength": "{:.3f}"})
        else:
            metric = "mean_return_log_score" if target == "Return density" else "mean_log_score"
            comparable = scores.dropna(subset=[metric])
            if target != "Return density":
                comparable = comparable.loc[comparable.score_group.eq("joint_core")]
            comparable = comparable.sort_values(metric)
            prefix = "return_score_" if target == "Return density" else "joint_score_"
            scope = comparable.iloc[0]
            st.caption(f"Common observations: {int(scope[prefix + 'days']):,} · "
                       f"{scope[prefix + 'start']} — {scope[prefix + 'end']} UTC")
            fig = px.bar(comparable, x=metric, y="model", orientation="h", color="model",
                         color_discrete_sequence=COLORS, text=metric,
                         labels={metric: "Mean log density", "model": ""})
            fig.update_traces(texttemplate="%{x:.4f}", textposition="outside")
            fig.update_layout(showlegend=False, hovermode="closest")
            chart(fig, max(270, len(comparable)*28+65))
        st.caption("Log score càng cao càng tốt. Core joint density dùng log return, realized volatility và relative volume.")
    with st.container(border=True, key="card_score_comparisons"):
        st.subheader("Score differences and 95% confidence intervals · " + config()["test_end"])
        comparison = summary("score_comparisons").query("asset == @asset").copy()
        comparison["left"] = comparison.left.fillna(comparison.model)
        comparison["right"] = comparison.right.fillna(comparison.reference)
        comparison["Chỉ tiêu"] = np.where(comparison.left.eq("HMM") & comparison.right.eq("GMM"),
                                           "Joint · core features", "Return")
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
        signal_chart(value, model, asset)


def portfolio(asset, model):
    performance, paths = portfolio_data(asset, model)
    cost = st.segmented_control("Transaction costs · bps", [5, 10, 25], default=10)
    cost = 10 if cost is None else cost
    view = performance.loc[performance.cost_bps.eq(cost)].copy()
    active = view.loc[view.strategy.eq("regime_aware")].iloc[0]
    grid_metrics([("CAGR · regime-aware", f"{active.cagr:.1%}"), ("Sharpe", f"{active.sharpe:.2f}"),
                  ("Maximum drawdown", f"{active.max_drawdown:.1%}"), ("Average exposure", f"{active.average_exposure:.1%}")])
    dates = paths[cost, "regime_aware"].index
    st.caption(f"{dates[0]:%d.%m.%Y} — {dates[-1]:%d.%m.%Y} UTC")
    for title, field, height in (("Portfolio value", "equity", 310), ("Drawdown", "drawdown", 245)):
        fig = go.Figure()
        for strategy, label in STRATEGIES.items():
            path = paths[cost, strategy]
            fig.add_trace(go.Scatter(x=path.index, y=path[field], name=label))
        with st.container(border=True, key="card_portfolio_" + field):
            st.subheader(f"{title} · {cost} bps")
            if field == "drawdown":
                fig.update_yaxes(tickformat=".0%")
            chart(fig, height)
    with st.container(border=True, key="card_portfolio_table"):
        st.subheader(f"Hiệu quả và rủi ro · {cost} bps")
        view["strategy"] = view.strategy.map(STRATEGIES)
        table(view[["strategy", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "turnover", "ending_value"]],
              {"cagr": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.3f}", "sortino": "{:.3f}",
               "max_drawdown": "{:.2%}", "turnover": "{:.2f}", "ending_value": "{:.3f}"})
        st.download_button("Tải hiệu quả danh mục", performance.to_csv(index=False), file_name=f"{asset}_{model}_portfolio.csv")
    with st.expander("Sharpe difference · 95% CI · " + config()["test_end"]):
        intervals = summary("portfolio_uncertainty").query("asset == @asset and model == @model")
        table(intervals[["sharpe_difference_ci_low", "sharpe_difference_ci_high", "block_days", "bootstrap_repetitions"]])


def states(asset, model):
    value, base = prediction(asset, model)
    store = Results()
    _, record = model_record(asset, model)
    diagnostics = model_diagnostics(asset, model)
    with st.container(border=True, key="card_state_signal"):
        st.subheader("Probabilities and signals · " + model)
        signal_chart(value, model, asset)
    transitions = [d for d in diagnostics if "transition" in d]
    left, right = st.columns(2)
    if transitions:
        latest = transitions[-1]
        matrix = np.asarray(latest["transition"])
        labels = [f"R{i+1}" for i in range(len(matrix))]
        if model == "MSGARCH":
            labels = [f"S{i+1} raw" for i in range(len(matrix))]
        with left, st.container(border=True, key="card_transition"):
            st.subheader("Transition matrix")
            st.caption(str(latest.get("period", latest.get("period_start")))[:10])
            names = [state_label(asset, model, i, k=len(matrix),
                     high_state=latest.get("high_unconditional_variance_state")) for i in range(len(matrix))]
            fig = px.imshow(matrix, x=labels, y=labels, text_auto=".2f", zmin=0, zmax=1,
                            color_continuous_scale=["#f1f7f6", "#147d78"],
                            labels={"x": "Next day", "y": "Current day", "color": "Probability"})
            fig.update_traces(customdata=[[f"{a}<br>→ {b}" for b in names] for a in names],
                              hovertemplate="%{customdata}<br>P=%{z:.3f}<extra></extra>")
            chart(fig, 325)
    with right if transitions else st.container(), st.container(border=True, key="card_regime_profile"):
        st.subheader("Regime profiles · 31.12.2024" if model == "HMM" else "Regime profiles")
        if model == "HMM":
            profiles = summary("regime_profiles").query("asset == @asset").drop(columns="asset").copy()
            profiles.insert(1, "Market regime", profiles.regime.map(
                lambda code: state_label(asset, model, int(code[1:])-1).split(" · ", 1)[-1]))
            table(profiles,
                  {"mean_return_daily": "{:.3%}", "volatility_annual": "{:.1%}", "occupancy": "{:.1%}",
                   "persistence": "{:.3f}", "expected_duration_days": "{:.1f}"})
            st.caption(label_note(model))
        elif "state" in value:
            counts = value.market_regime.value_counts(normalize=True).rename_axis("Market regime").reset_index(name="Observation share")
            table(counts, {"Observation share": "{:.1%}"})
        elif model == "BOCPD":
            table(pd.DataFrame([record["decision"]]))
        if model != "HMM":
            st.caption(label_note(model))
    coefficient_path = base + ("_coefficients.csv" if base.startswith("extensions/") else ".coefficients.csv")
    if store.exists(coefficient_path):
        with st.container(border=True, key="card_coefficients"):
            st.subheader("Model coefficients")
            coefficients = read_table(coefficient_path)
            st.caption("Đến " + str(coefficients.period.max())[:10])
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
                rows.append({"Tháng": fit["period"][:10], "Raw state": f"S{state_id+1}",
                    "Market regime": state_label(asset, model, state_id,
                        high_state=fit["high_unconditional_variance_state"]).split(" · ", 1)[1],
                    "alpha": weights[state_id, 0], "beta": weights[state_id, 1],
                    "omega": variance[state_id]*weights[state_id, 2], "Annualized sigma": np.sqrt(365*variance[state_id])})
        with st.container(border=True, key="card_garch_parameters"):
            st.subheader("GARCH parameters")
            table(pd.DataFrame(rows))


def data(asset, model):
    market = market_data(asset)
    sources = data_sources()
    report = ROOT / "results/tom_tat_ket_qua.html"
    if report.exists():
        st.download_button("Tải bản tổng hợp kết quả", report.read_bytes(), file_name=report.name, mime="text/html")
    grid_metrics([("Bắt đầu dữ liệu spot", "17.08.2017"), ("Ngày dữ liệu mới nhất", f"{market.index[-1]:%d.%m.%Y}"),
                  ("Tài sản", "BTC / ETH"), ("Chuỗi vĩ mô / on-chain", str(sources.series.notna().sum()))])
    with st.container(border=True, key="card_data_explorer"):
        st.subheader("Khám phá dữ liệu")
        series = st.selectbox("Chỉ số", list(SERIES), format_func=SERIES.get)
        if series in market:
            chart(px.line(market, x=market.index, y=series, labels={series: SERIES[series], "date": "Ngày UTC"}), 360)
        else:
            panel = panel_data(asset)
            if series in panel and panel[series].notna().any():
                chart(px.line(panel, x=panel.index, y=series, labels={series: SERIES[series], "date": "Ngày khả dụng · UTC"}), 360)
            else:
                st.info("Đang chờ dữ liệu từ nguồn.")
    with st.container(border=True, key="card_data_sources"):
        st.subheader("Nguồn và độ phủ dữ liệu")
        source_view = sources.copy()
        source_view["Chỉ số"] = source_view.series.fillna(source_view.dataset).map(lambda v: SERIES.get(v, {"spot": "Spot OHLCV"}.get(v, v)))
        source_view["Nguồn"] = source_view.source.fillna("Binance")
        source_view["Số quan sát"] = source_view.observations.fillna(source_view.valid_days)
        for col in ("first_date", "last_date"):
            source_view[col] = pd.to_datetime(source_view[col], utc=True, format="mixed").dt.strftime("%d.%m.%Y")
        table(source_view[["Chỉ số", "asset", "Nguồn", "first_date", "last_date", "Số quan sát"]])
    with st.expander("Trạng thái cập nhật"):
        cache = runtime().store
        rows = []
        for name in summary("model_scores").query("asset == @asset").model:
            record = cache.json(f"model:{asset}:{name}")
            rows.append({"Mô hình": name, "Ngày kết quả UTC": record.get("through", config()["test_end"]),
                         "Trạng thái": "Đang chạy" if record.get("running") else record.get("error") or "Sẵn sàng"})
        table(pd.DataFrame(rows))
        for source, error in cache.json("external_status").get("errors", {}).items():
            st.write(source + ": " + error)
        fred = cache.json("fred_daily")
        if fred.get("day"):
            st.caption("FRED · một đợt mỗi ngày UTC · ngày đã thử: " + fred["day"])
    with st.container(border=True, key="card_sample_counts"):
        st.subheader("Cỡ mẫu theo nhóm biến")
        counts = summary("sample_counts").query("asset == @asset")
        table(counts[["cohort", "split", "start", "end", "complete_feature_days"]])
    with st.container(border=True, key="card_methods"):
        st.subheader("Phương pháp nghiên cứu")
        cfg = config()
        st.markdown(f"Dữ liệu ngày UTC; log return, rolling volatility, relative volume, liquidity, "
                    f"funding/OI và các biến vĩ mô/on-chain. Train đến **{cfg['train_end']}**; "
                    f"validation **{cfg['validation_start']} – {cfg['validation_end']}**; "
                    f"đánh giá **{cfg['test_start']} – {cfg['test_end']}**.")
        st.markdown("Mô hình gồm HMM, GMM, K-means, Markov switching, TVTP, MS-GARCH, BOCPD và HDBSCAN. "
                    "Các mô hình tái ước lượng dùng expanding window theo tháng; BOCPD cập nhật tuần tự. "
                    "Predictive density được tính trước khi quan sát ngày cần dự báo.")
        st.markdown("Danh mục Buy-and-hold, Volatility targeting và Regime-aware; volatility target 40%/năm, "
                    "exposure 0–100%, lãi tiền mặt 0. Tín hiệu cuối ngày t giao dịch ở open t+2. "
                    "Chi phí 5/10/25 bps; bootstrap block 30 ngày, 1.000 lần để tính 95% confidence intervals.")


PAGES = {"Tổng quan": (":material/space_dashboard:", overview), "Mô hình": (":material/analytics:", models),
         "Danh mục": (":material/account_balance_wallet:", portfolio), "Trạng thái": (":material/hub:", states),
         "Dữ liệu": (":material/database:", data)}
