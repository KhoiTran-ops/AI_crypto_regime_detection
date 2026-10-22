"""Research dashboard: published findings, no model fitting during navigation."""
from pathlib import Path

import streamlit as st

from regime.dashboard_views import PAGES, read_json, summary
from regime.live_view import live_view, runtime

PAGES = {"Trực tiếp": (":material/sensors:", live_view), **PAGES}

st.set_page_config(page_title="Crypto Regime · Nghiên cứu", page_icon="◈", layout="wide")
st.markdown("<style>" + (Path(__file__).with_suffix(".css")).read_text(encoding="utf-8") + "</style>",
            unsafe_allow_html=True)


def navigate(page):
    st.session_state.page = page


if "page" not in st.session_state:
    st.session_state.page = "Trực tiếp"

with st.sidebar:
    for label, (icon, _) in PAGES.items():
        st.button(label, icon=icon, key="nav_" + label, width="stretch",
                  type="primary" if st.session_state.page == label else "secondary",
                  on_click=navigate, args=(label,))

try:
    run = read_json("run.json")
    scores = summary("model_scores")
except (FileNotFoundError, KeyError):
    st.error("Chưa tìm thấy gói kết quả nghiên cứu. Xem hướng dẫn cài đặt trong README.")
    st.stop()

page = st.session_state.page
heading, controls = st.columns([3, 2], vertical_alignment="center")
with heading:
    st.markdown('<div class="eyebrow">CRYPTO MARKET REGIME DETECTION</div>', unsafe_allow_html=True)
    st.title(page)
with controls:
    left, right = st.columns(2)
    asset = left.selectbox("Tài sản", run["assets"], format_func=lambda v: v.replace("USDT", " / USDT"), key="asset")
    options = scores.loc[scores.asset.eq(asset), "model"].tolist()
    if page == "Danh mục":
        available = summary("portfolio_performance")
        options = available.loc[available.asset.eq(asset), "model"].drop_duplicates().tolist()
    if "HMM" in options:
        options = ["HMM", *[m for m in options if m != "HMM"]]
    model = None
    if page != "Dữ liệu":
        if st.session_state.get("model") not in options:
            st.session_state.model = "HMM"
        model = right.selectbox("Mô hình", options, key="model")
    else:
        right.markdown('<div class="snapshot">SNAPSHOT<br><strong>30.09.2026</strong></div>', unsafe_allow_html=True)

st.markdown('<div class="page-rule"></div>', unsafe_allow_html=True)
runtime()
PAGES[page][1](asset, model)
