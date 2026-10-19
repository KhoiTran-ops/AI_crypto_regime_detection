"""Supplemental evidence UI; numerical work stays in the offline pipeline."""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .data import ROOT


def render_extensions(asset, chart):
    folder = ROOT / "results/extensions"
    st.markdown("### Dữ liệu bổ sung và các đối chứng mới")
    st.caption("Phân tích hồi cứu. Giữ cách chia mẫu và ngưỡng RMSE của bản gốc; không chọn lại theo kết quả test.")
    coverage = pd.read_csv(folder / "source_coverage.csv")
    st.dataframe(coverage, hide_index=True, width="stretch")
    st.caption("BTC/ETH price và volume dùng Binance. Coin Metrics chỉ bổ sung vốn hóa; DXY là chỉ số ICE qua Yahoo. "
               "Stablecoin/TVL theo phạm vi theo dõi của DefiLlama, không bảo đảm toàn bộ thị trường.")
    panel_path = folder / "external_panel.parquet"
    if not panel_path.exists():
        st.info("Bảng kết quả xem được ngay. Để xem chuỗi dữ liệu bổ sung, tải dữ liệu miễn phí về máy; không cần API key.")
        if st.button("Tải dữ liệu bổ sung", key="load_external"):
            from .external import build_external
            with st.spinner("Đang tải và kiểm tra dữ liệu…"):
                from .supply import add_supply
                add_supply(build_external())
            st.rerun()
    else:
        panel = pd.read_parquet(panel_path)
        names = {"vix": "VIX", "dxy": "DXY", "ust10": "Treasury 10 năm (%)", "ust2": "Treasury 2 năm (%)",
                 "sp500": "S&P 500 price index", "stablecoin_cap": "Stablecoin USD", "defi_tvl": "DeFi TVL (USD)",
                 "btc_market_cap": "Vốn hóa BTC (USD)", "eth_market_cap": "Vốn hóa ETH (USD)"}
        if "stablecoin_supply" in panel:
            names["stablecoin_supply"] = "Cung stablecoin theo mệnh giá USD"
        series = st.selectbox("Chuỗi bổ sung", list(names), format_func=names.get)
        chart(px.line(panel.reset_index(), x="date", y=series, labels={series: names[series], "date": "Ngày được phép dùng"}), 300)
        st.caption("Biểu đồ theo ngày khả dụng giả định. Mô hình còn dùng độ trễ hai ngày; lịch công bố lịch sử chưa được xác nhận đầy đủ.")
    if not (folder / "run.json").exists():
        st.info("Pipeline mở rộng chưa hoàn tất; xem kết quả gốc ở các tab khác.")
        return
    scores = pd.read_csv(folder / "model_scores.csv").query("asset == @asset")
    st.markdown("**Đánh giá mô hình**")
    st.dataframe(scores, hide_index=True, width="stretch")
    st.caption("Log score lớn hơn là tốt hơn cho cùng biến return. HDBSCAN không có mật độ dự báo; membership strength không phải xác suất stress.")
    st.dataframe(pd.read_csv(folder / "score_comparisons.csv").query("asset == @asset"), hide_index=True, width="stretch")
    model = st.selectbox("Xem diễn biến mô hình bổ sung", scores.model.tolist(), key="extension_model")
    prediction = pd.read_parquet(folder / f"{asset}_{model}.parquet")
    if "p_high_vol" in prediction:
        chart(px.line(prediction.reset_index(), x="date", y="p_high_vol",
                      labels={"p_high_vol": "Xác suất trạng thái có phương sai cao", "date": "Ngày"}), 280)
    elif "change_probability" in prediction:
        chart(px.line(prediction.reset_index(), x="date", y="change_probability",
                      labels={"change_probability": "Xác suất điểm thay đổi BOCPD", "date": "Ngày"}), 280)
        st.caption("Điểm thay đổi cấu trúc khác với trạng thái stress của HMM.")
    else:
        chart(px.scatter(prediction.reset_index(), x="date", y="state", color="noise",
                         labels={"state": "Cụm; -1 là noise", "date": "Ngày"}), 280)
    coefficient = folder / f"{asset}_{model}_coefficients.csv"
    if coefficient.exists():
        st.markdown("**Hệ số chuyển trạng thái theo từng lần ước lượng**")
        details = json.loads((folder / f"{asset}_{model}.json").read_text(encoding="utf-8"))
        mapping = pd.DataFrame(details["diagnostics"])[["period", "raw_high_vol_state"]]
        st.dataframe(pd.read_csv(coefficient).merge(mapping, on="period", validate="many_to_one"),
                     hide_index=True, width="stretch")
        st.caption("Dấu hệ số phải đọc cùng trạng thái đích trong diagnostics; không diễn giải là tác động nhân quả.")
    performance = pd.read_csv(folder / "portfolio_performance.csv")
    if model in performance.model.unique():
        cost = st.select_slider("Chi phí mô phỏng bổ sung", options=[5, 10, 25], value=10, key="extension_cost")
        st.dataframe(performance[performance.asset.eq(asset) & performance.model.eq(model) & performance.cost_bps.eq(cost)],
                     hide_index=True, width="stretch")
        fig = go.Figure()
        for strategy in ("buy_hold", "vol_target", "regime_aware"):
            path = pd.read_parquet(folder / f"{asset}_{model}_{strategy}_portfolio.parquet")
            fig.add_trace(go.Scatter(x=path.index, y=path.equity, name=strategy))
        chart(fig, 300)
        st.caption("Đường tài sản tại mức phí chính 10 bps; bảng phía trên theo phí đang chọn.")
    st.markdown("**Thanh khoản và tương quan theo trạng thái HMM gốc**")
    st.dataframe(pd.read_csv(folder / "liquidity_profiles.csv").query("asset == @asset"), hide_index=True, width="stretch")
    st.download_button("Tải bảng kết quả bổ sung", (folder / "model_scores.csv").read_bytes(),
                       file_name="extension_model_scores.csv", mime="text/csv")
    st.caption("Chi tiết nguồn, giả định và lựa chọn tham số: docs/EXTERNAL_METHODS.md. Bài cũ chưa cập nhật phần mở rộng này.")
