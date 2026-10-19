"""Vietnamese research dashboard; reads published artifacts and never refits on interaction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COLORS = ["#3566A8", "#237E83", "#9F781F", "#B04D2C", "#A94456"]
STRATEGIES = {"buy_hold": "Mua và giữ", "vol_target": "Điều chỉnh theo biến động",
              "regime_aware": "Điều chỉnh theo trạng thái"}

st.set_page_config(page_title="Crypto Regime Radar", page_icon="◉", layout="wide")
st.markdown('<style>[data-testid="stCaptionContainer"]{color:#53647B}</style>', unsafe_allow_html=True)


def read_table(name):
    path = RESULTS / name
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def chart(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=12, r=16, t=35, b=12),
                      font=dict(family="Arial", size=13), hovermode="x unified",
                      legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, width="stretch")


if not (RESULTS / "run.json").exists():
    st.info("Chưa có bộ kết quả hoàn tất. Chạy quy trình nghiên cứu theo README trước khi mở dashboard.")
    st.stop()

run = read_json("run.json")
with st.sidebar:
    st.markdown("## ◉ Regime Radar")
    asset = st.selectbox("Tài sản", run["assets"], format_func=lambda item: item.replace("USDT", " / USDT"))
    st.caption("Binance và nguồn bổ sung · ngày UTC")
    st.divider()
    st.markdown("**Phạm vi đánh giá cuối**")
    st.write("01/01/2025 – 30/09/2026")
    st.caption("Phần mở rộng là đánh giá hồi cứu; kỳ test đã được quan sát.")
    report = ROOT / "paper/bao_cao_nghien_cuu.md"
    if report.exists():
        st.download_button("Tải bản thảo nghiên cứu", report.read_bytes(),
                           file_name="bao_cao_nghien_cuu.md", mime="text/markdown")
    st.download_button("Tải bảng hiệu quả danh mục",
                       (RESULTS / "portfolio_performance.csv").read_bytes(),
                       file_name="portfolio_performance.csv", mime="text/csv")
    st.caption("Mô phỏng nghiên cứu; trạng thái ẩn không phải nhãn thị trường đã được xác thực.")

frame = read_table(f"{asset}_market.parquet")
prediction = read_table(f"{asset}_HMM.parquet")
diagnostics = read_json(f"{asset}_HMM.diagnostics.json")
last_fit = diagnostics[-1]
probability_columns = [column for column in prediction if column.startswith("p_") and column != "p_high_vol"]
k = len(probability_columns)
last = prediction.iloc[-1]
latest_date = prediction.index[-1]
market = frame.loc[latest_date]

st.title("Crypto Market Regime Radar")
st.caption(f"{asset.replace('USDT', '/USDT')} · Kết quả ngoài mẫu · Chốt ngày {latest_date:%d/%m/%Y} UTC")
a, b, c, d = st.columns(4)
a.metric("Giá đóng cửa · USDT", f"{market.close:,.2f}", f"{np.expm1(market['return']):+.2%}")
b.metric("Trạng thái có xác suất cao nhất", f"R{int(last.state)+1}")
b.caption(f"{last[probability_columns].max():.1%} xác suất")
c.metric("Xác suất trạng thái σ cao nhất", f"{last.p_high_vol:.1%}")
d.metric("Biến động 30 ngày · năm hóa", f"{market.rv30:.1%}")
st.caption(f"Xác suất trạng thái σ cao nhất ở ngày tiếp theo: {last.next_high_vol:.1%}. "
           "Các mã R được ghép theo đặc trưng qua từng lần ước lượng; không áp đặt nhãn bull/bear.")

tabs = st.tabs(["Diễn biến", "Mô hình", "Danh mục & rủi ro", "Chuyển trạng thái",
                "Funding & OI", "BTC / ETH", "Dữ liệu & phương pháp", "Dữ liệu & mô hình mở rộng"])

with tabs[0]:
    selected = st.date_input("Khoảng thời gian", (prediction.index[0].date(), latest_date.date()),
                            min_value=prediction.index[0].date(), max_value=latest_date.date())
    if len(selected) == 2:
        left, right = selected
    else:
        left, right = prediction.index[0].date(), latest_date.date()
    visible = prediction.loc[str(left):str(right)]
    joined = visible.join(frame[["close"]])
    fig = go.Figure(go.Scatter(x=joined.index, y=joined.close, name="Giá đóng cửa",
                               line=dict(color="#354258", width=1.5)))
    for state in range(k):
        points = joined[joined.state.eq(state)]
        fig.add_trace(go.Scatter(x=points.index, y=points.close, name=f"R{state+1}",
                                mode="markers", marker=dict(size=4, color=COLORS[state])))
    fig.update_yaxes(title_text="USDT")
    chart(fig, 410)
    fig = go.Figure()
    for state, column in enumerate(probability_columns):
        fig.add_trace(go.Scatter(x=visible.index, y=visible[column], stackgroup="one",
                                name=f"R{state+1}", line=dict(width=.5, color=COLORS[state])))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Xác suất lọc")
    chart(fig, 290)
    st.caption("Xác suất lọc dùng thông tin đến cuối ngày tương ứng. Mô hình được ước lượng lại đầu tháng bằng lịch sử trước tháng đó.")

with tabs[1]:
    scores = read_table("model_scores.csv").query("asset == @asset")
    group = st.selectbox("Nhóm mật độ dự báo có thể so sánh", ["joint_core", "return_only_A", "return_only_B"],
                         format_func=lambda value: {"joint_core": "Return + biến động + khối lượng tương đối",
                          "return_only_A": "Chỉ return: HMM và Markov Switching AR",
                          "return_only_B": "Chỉ return: chuyển cố định và chuyển theo funding/volume"}[value])
    comparable = scores[scores.score_group.eq(group)]
    chart(px.bar(comparable, x="model", y="mean_log_score", color="model",
                 labels={"model": "Mô hình", "mean_log_score": "Log density trung bình"}), 300)
    st.dataframe(scores, hide_index=True, width="stretch")
    st.caption("Log density lớn hơn là tốt hơn trong cùng nhóm biến. K-means không có mật độ dự báo. "
               "Mean return log score của HMM là mật độ return trước khi quan sát dữ liệu ngày đó.")
    st.markdown("**Lựa chọn số trạng thái trên validation**")
    selection = read_table("model_selection.csv").query("asset == @asset")
    st.dataframe(selection, hide_index=True, width="stretch")
    st.markdown("**Bất định của chênh lệch mật độ dự báo**")
    st.dataframe(read_table("score_comparisons.csv").query("asset == @asset"),
                 hide_index=True, width="stretch")
    st.caption("Khoảng tin cậy bootstrap theo block 30 ngày, 1.000 lần; dương nghiêng về mô hình ở cột left.")

with tabs[2]:
    cost = st.select_slider("Chi phí giả định trên giá trị giao dịch", options=[5, 10, 25], value=10,
                            format_func=lambda value: f"{value} bps")
    performance = read_table("portfolio_performance.csv")
    view = performance.query("asset == @asset and model == 'HMM' and cost_bps == @cost").copy()
    view["strategy"] = view.strategy.map(STRATEGIES)
    st.dataframe(view[["strategy", "cagr", "volatility", "sharpe", "max_drawdown", "turnover", "ending_value"]]
                 .style.format({"cagr": "{:.1%}", "volatility": "{:.1%}", "sharpe": "{:.2f}",
                                "max_drawdown": "{:.1%}", "turnover": "{:.1f}", "ending_value": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("Đường tài sản và drawdown dưới đây dùng mức chi phí chính 10 bps.")
    equity, drawdown = go.Figure(), go.Figure()
    for strategy, label in STRATEGIES.items():
        path = read_table(f"{asset}_HMM_{strategy}_portfolio.parquet")
        equity.add_trace(go.Scatter(x=path.index, y=path.equity, name=label))
        drawdown.add_trace(go.Scatter(x=path.index, y=path.drawdown, name=label))
    equity.update_yaxes(title_text="Giá trị ban đầu = 1")
    drawdown.update_yaxes(tickformat=".0%", title_text="Drawdown")
    chart(equity)
    chart(drawdown, 270)
    st.dataframe(read_table("portfolio_uncertainty.csv").query("asset == @asset and model == 'HMM'"),
                 hide_index=True, width="stretch")
    st.caption("Mục tiêu biến động 40%/năm, tỷ trọng 0–100%, tiền mặt lãi suất 0. "
               "Tín hiệu cuối ngày t giao dịch tại open t+2; thanh lý tại open cuối kỳ. "
               "Khoảng tin cậy Sharpe chứa 0 không hỗ trợ khẳng định cải thiện rõ ràng.")

with tabs[3]:
    transition = np.array(last_fit["transition"])
    labels = [f"R{i+1}" for i in range(k)]
    chart(px.imshow(transition, x=labels, y=labels, text_auto=".2f", zmin=0, zmax=1,
                    color_continuous_scale="Blues",
                    labels={"x": "Trạng thái ngày kế tiếp", "y": "Trạng thái hiện tại", "color": "Xác suất"}), 390)
    st.caption(f"Ma trận từ lần ước lượng {last_fit['period_start'][:10]}, dữ liệu fit đến {last_fit['train_end'][:10]}.")
    profile = pd.DataFrame(read_json(f"{asset}_initial_states.json")["states"])
    profile["regime"] = profile.state.map(lambda value: f"R{value+1}")
    st.dataframe(profile, hide_index=True, width="stretch")
    st.caption("Bảng đặc trưng lấy từ mô hình đầu test, fit đến 31/12/2024. Thời lượng kỳ vọng là 1/(1-pii), "
               "có thể bất định lớn khi pii gần 1; không coi đây là thời lượng đã biết của thị trường.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction.index, y=prediction.p_high_vol, name="Xác suất lọc hôm nay"))
    fig.add_trace(go.Scatter(x=prediction.index, y=prediction.next_high_vol, name="Xác suất ngày kế tiếp"))
    fig.update_yaxes(tickformat=".0%")
    chart(fig, 300)

with tabs[4]:
    derivative = frame.loc["2019-11-28":]
    chart(px.line(derivative, x=derivative.index, y="funding",
                  labels={"funding": "Tổng funding trong ngày", "x": "Ngày UTC"}), 300)
    chart(px.line(derivative, x=derivative.index, y="oi",
                  labels={"oi": "OI cuối ngày · đơn vị tài sản", "x": "Ngày UTC"}), 300)
    st.caption("Funding/OI thuộc hợp đồng perpetual USD-M, không phải thị trường spot. "
               "Các ngày OI không có snapshot hợp lệ trong 15 phút cuối ngày được giữ thiếu.")
    st.markdown("**Mô hình chuyển trạng thái theo funding và khối lượng**")
    coefficients = read_table(f"{asset}_NHHMM.coefficients.csv")
    latest_period = coefficients.period.max()
    st.dataframe(coefficients[(coefficients.period.eq(latest_period)) &
                              coefficients.parameter.str.contains("tvtp")],
                 hide_index=True, width="stretch")
    driver_diagnostics = read_json(f"{asset}_NHHMM.diagnostics.json")[-1]
    st.caption(f"Trạng thái σ cao nhất có mã thô {driver_diagnostics['raw_high_vol_state']}. "
               "tvtp0: hằng số; tvtp1: funding trễ 2 ngày; tvtp2: volume tương đối trễ 2 ngày, "
               "hai biến được chuẩn hóa bằng train. p[i→0] là log odds chuyển sang trạng thái thô 0 so với 1. "
               "Hệ số là quan hệ thống kê, không chứng minh quan hệ nhân quả.")
    st.write("Hội tụ lần ước lượng cuối:", "Đạt" if driver_diagnostics["converged"] else "Chưa đạt")
    st.dataframe(read_table("cohort_selection.csv").query("asset == @asset"),
                 hide_index=True, width="stretch")

with tabs[5]:
    prices, risks = go.Figure(), go.Figure()
    for symbol in run["assets"]:
        other = read_table(f"{symbol}_market.parquet").loc[run["test_start"]:run["cutoff"]]
        probs = read_table(f"{symbol}_HMM.parquet")
        prices.add_trace(go.Scatter(x=other.index, y=100 * other.close / other.close.iloc[0],
                                   name=symbol.replace("USDT", "")))
        risks.add_trace(go.Scatter(x=probs.index, y=probs.p_high_vol, name=symbol.replace("USDT", "")))
    prices.update_yaxes(title_text="Chỉ số giá · đầu kỳ = 100")
    risks.update_yaxes(title_text="P(trạng thái σ cao nhất)", tickformat=".0%")
    chart(prices)
    chart(risks)
    st.caption("Mỗi tài sản có mô hình và phân phối trạng thái riêng; mã R của BTC không tương đương trực tiếp mã R của ETH.")

with tabs[6]:
    with st.expander("Lịch sử giá trong mẫu phân tích từ 2017"):
        history = go.Figure(go.Scatter(x=frame.index, y=frame.close, name="Giá đóng cửa",
                                      line=dict(color=COLORS[0], width=1)))
        for boundary in ("2023-01-01", "2025-01-01"):
            history.add_vline(x=boundary, line_dash="dash", line_color="#61718A")
        history.update_yaxes(type="log", title_text="Giá USDT · thang log")
        chart(history, 380)
        st.caption("Hai vạch dọc là đầu validation 2023 và đầu holdout 2025 của mẫu A. "
                   "Biểu đồ này mô tả giá toàn mẫu; các đồ thị regime phía trước dùng đánh giá 2025–2026.")
    st.markdown("**Nguồn và chất lượng**")
    st.dataframe(read_table("data_quality.csv"), hide_index=True, width="stretch")
    st.markdown("**Mẫu dùng trong nghiên cứu**")
    st.dataframe(read_table("sample_counts.csv"), hide_index=True, width="stretch")
    st.markdown("**Kiểm chứng cỡ mẫu**")
    st.dataframe(read_table("simulation_summary.csv"), hide_index=True, width="stretch")
    st.caption("Ngưỡng của dự án: RMSE chuyển trạng thái ≤0,05; RMSE tương đối volatility ≤20%; "
               "hội tụ ≥90%. Mô phỏng dựa trên Gaussian DGP từ train, không xác nhận trạng thái thật.")
    st.markdown("**Giới hạn cần đọc cùng kết quả**")
    st.write("• Hai cặp spot BTC/USDT và ETH/USDT trên Binance, không đại diện toàn thị trường.\n"
             "• Archive tải hồi cứu không cung cấp đầy đủ phiên bản dữ liệu theo thời điểm công bố.\n"
             "• Rolling volatility tạo phụ thuộc; Gaussian diagonal là xấp xỉ.\n"
             "• Chi phí 5/10/25 bps là kịch bản, chưa mô phỏng sổ lệnh hay market impact.\n"
             "• Funding/OI là mở rộng khám phá trên mẫu ngắn hơn.\n"
             "• Lựa chọn K chạm biên lưới không chứng minh đó là số trạng thái thật.")
    if (RESULTS / "revision.json").exists():
        with st.expander("Nhật ký sửa lỗi tái lập"):
            st.write("v2 truyền seed trực tiếp cho Generator của statsmodels. Lượt đánh giá v1 đã xem holdout; "
                     "v2 là tái tính sửa lỗi có lưu lịch sử, không phải một holdout mới chưa từng được xem.")
            st.json(read_json("revision.json"))
    with st.expander("Thiết kế đã khóa"):
        st.json(read_json("protocol_lock.json"))
    st.caption(f"Mã phiên bản nghiên cứu: {run['protocol_hash'][:16]}")


with tabs[7]:
    from regime.extension_dashboard import render_extensions
    render_extensions(asset, chart)
