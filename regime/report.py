"""One offline HTML research summary, generated solely from frozen results."""
from hashlib import sha256
from html import escape
from io import StringIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .data import ROOT
from .labels import LABEL_NOTE, state_label
from .results import Results

ASSETS = ("BTCUSDT", "ETHUSDT")
COLORS = ("#147d78", "#4263eb", "#c98a25", "#b85651", "#875db2")
STRATEGIES = {"buy_hold": "Buy-and-hold", "vol_target": "Volatility targeting", "regime_aware": "Regime-aware"}
HEADERS = {"asset": "Tài sản", "model": "Mô hình", "regime": "Mã", "cohort": "Nhóm biến",
           "split": "Tập mẫu", "start": "Từ", "end": "Đến", "complete_feature_days": "Ngày hợp lệ",
           "mean_return_daily": "Mean daily log return", "volatility_annual": "Annualized volatility", "occupancy": "Posterior-weighted occupancy",
           "persistence": "Persistence", "expected_duration_days": "Expected duration (days)",
           "test_days": "Ngày test", "mean_return_log_score": "Return log score", "mean_log_score": "Joint log score",
           "noise_fraction": "Noise fraction", "mean_membership_strength": "Membership strength", "strategy": "Chiến lược",
           "volatility": "Annualized volatility", "max_drawdown": "Maximum drawdown", "average_exposure": "Average exposure",
           "cost_bps": "Phí (bps)", "mean_difference": "Chênh lệch", "ci_low": "95% CI lower",
           "ci_high": "95% CI upper", "n_common": "Ngày chung", "validation_log_score": "Validation log score",
           "train_bic": "Training BIC", "train_aic": "Training AIC", "n_parameters": "Parameter count", "state_changes": "State changes"}
SOURCES = (
    ("Hyndman & Athanasopoulos · rolling forecasting origin", "https://otexts.com/fpp3/toolbox.html#time-series-cross-validation"),
    ("Cawley & Talbot (2010) · selection bias", "https://jmlr.org/papers/v11/cawley10a.html"),
    ("ECB (23.06.2023) · crypto winter 2021–2022", "https://www.ecb.europa.eu/press/key/date/2023/html/ecb.sp230623_1~80751450e6.en.html"),
    ("SEC (10.01.2024) · spot Bitcoin ETP", "https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023"),
    ("Pohle et al. (2017) · số trạng thái HMM", "https://arxiv.org/abs/1701.08673"),
    ("Morris et al. (2019) · thiết kế mô phỏng và MCSE", "https://doi.org/10.1002/sim.8086"),
    ("Koki et al. (2022) · diễn giải regime crypto", "https://arxiv.org/html/2011.03741"),
    ("Cortese et al. (2023) · bull, neutral, bear", "https://link.springer.com/article/10.1007/s42521-023-00085-x"),
)


def table(frame, formats=None):
    value = frame.copy()
    for column, spec in (formats or {}).items():
        value[column] = value[column].map(lambda x: format(x, spec) if pd.notna(x) else "—")
    return '<div class="table">' + value.rename(columns=HEADERS).to_html(
        index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.5f}", escape=True) + '</div>'


def figure(fig, caption):
    output = StringIO()
    fig.savefig(output, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = " " + " ".join(output.getvalue().split("<svg", 1)[1].split())
    return '<figure><svg' + svg + '<figcaption>' + escape(caption) + '</figcaption></figure>'


def build_report():
    store = Results()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "crypto-study"})
    sections = []
    def add(title, content):
        sections.append('<section><h2>' + escape(title) + '</h2>' + content + '</section>')

    scores = store.table("model_scores")
    performance = store.table("portfolio_performance")
    profiles = store.table("regime_profiles")
    profiles.insert(2, "Market regime", [state_label(a, "HMM", int(r[1:])-1).split(" · ", 1)[1]
                                               for a, r in zip(profiles.asset, profiles.regime, strict=True)])
    counts = store.table("sample_counts")
    core = counts.loc[counts.cohort.eq("A"), ["asset", "split", "start", "end", "complete_feature_days"]]
    paragraph = ("<p><b>Dữ liệu và biến.</b> BTCUSDT/ETHUSDT dùng OHLCV spot Binance từ 17.08.2017 đến 30.09.2026 UTC; "
                 "bỏ ngày niêm yết không trọn vẹn, mẫu phân tích bắt đầu 17.10.2017 sau warm-up của panel có cửa sổ tối đa 60 ngày. "
                 "Biến lõi: log(Ct/Ct−1), rv30 = sample SD của daily log return trong 30 ngày × √365, volume_z = log(1+khối lượng USDT) "
                 "standardization bằng mean/SD của 30 ngày trước. Funding/OI Binance và 10 chuỗi VIX, DXY, Treasury 2/10 năm, "
                 "S&amp;P500, BTC/ETH market cap, stablecoin market cap/supply, DeFi TVL bổ sung thông tin. Không dùng giá BTC/ETH ngoài Binance. "
                 "Mỗi nhóm biến dùng lịch khả dụng riêng; dữ liệu Mỹ giả định khả dụng ngày làm việc Mỹ kế tiếp, tổng hợp crypto ngày UTC kế tiếp, "
                 "biến chuyển trạng thái trễ thêm 2 ngày; có kiểm tra macro trễ thêm 7 ngày. Chuẩn hóa mô hình chỉ fit trên quá khứ.</p>")
    add("1. Dữ liệu và thiết kế mẫu", paragraph + table(core))
    add("Cơ sở chọn giai đoạn", "<p>Chia theo thời gian và rolling origin phù hợp dự báo chuỗi thời gian [1]; tách lựa chọn mô hình khỏi "
        "đánh giá giúp giảm selection bias [2]. Train đến 2022 bao quát nhiều pha và đợt crypto winter 2021–2022 được ECB ghi nhận [3]; "
        "validation 2023–2024 có bối cảnh khác, gồm việc SEC chấp thuận spot Bitcoin ETP ngày 10.01.2024 [4]. Đây là diễn giải kinh tế "
        "cho các giai đoạn, không chứng minh ngày 31.12.2022/31.12.2024 tối ưu hoặc là statistical structural break. Mốc năm tròn là lựa chọn thiết kế, "
        "không phải tỷ lệ chia chuẩn bắt buộc. Test gồm 638 ngày tiếp theo đến snapshot hoàn chỉnh; các phân tích sau khi đã xem test là hồi cứu.</p>")
    split_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    for asset, ax in zip(ASSETS, axes, strict=True):
        market = store.frame(f"{asset}_market.parquet")
        price = market.close
        ax.plot(price.index, 100*price/price.iloc[0], color=COLORS[0], lw=1)
        for boundary in ("2023-01-01", "2025-01-01"):
            ax.axvline(pd.Timestamp(boundary, tz="UTC"), color="#9aa6b2", ls="--", lw=.9)
        ax.set(title=asset[:3], ylabel="Normalized price, initial = 100", yscale="log")
        for row in core.loc[core.asset.eq(asset)].itertuples():
            sample = market.loc[row.start:row.end]
            split_rows.append({"asset": asset, "split": row.split, "Ngày": len(sample),
                               "Mean daily log return": sample["return"].mean(),
                               "Annualized realized volatility": sample["return"].std(ddof=1)*np.sqrt(365)})
    fig.tight_layout()
    add("Đặc trưng từng giai đoạn", table(pd.DataFrame(split_rows), {"Mean daily log return": ".3%", "Annualized realized volatility": ".1%"})
        + figure(fig, "Hình 1. Giá đóng cửa chuẩn hóa; đường đứt: đầu validation và đầu test. Số liệu theo mẫu mang tính mô tả."))
    origins = pd.concat([store.frame(f"{a}_nested_origins.csv").assign(asset=a) for a in ASSETS], ignore_index=True)
    add("Kiểm tra cuốn chiếu đã chạy", table(origins[["asset", "outer_year", "inner_validation_year", "chosen_k_from_past_only", "n_outer_train", "outer_log_score"]])
        + "<p>K thay đổi theo origin; nested origins không áp recovery gate giống lựa chọn cuối, nên không xác nhận riêng BTC K=4/ETH K=5 hay một ngày cắt tối ưu.</p>")
    methods = ("<p><b>Phương pháp.</b> Gaussian HMM với diagonal covariance dùng EM và forward filtering P(St|Y1:t); chọn K trong 2–5 bằng "
               "log density validation kèm kiểm chứng recovery hiện hành. Sau lựa chọn, fit đầu test trên 2.633 ngày đến 31.12.2024, "
               "rồi refit hàng tháng với expanding window chỉ dùng quá khứ; ghép trạng thái giữa các lần fit bằng Hungarian theo mean/SD. "
               "Đối chiếu GMM, K-means, HMM return/funding/OI, Markov switching/AR, TVTP với đối chứng MS0 cùng mẫu, "
               "MS-GARCH, BOCPD và HDBSCAN. Predictive density tính trước quan sát ngày dự báo; chỉ so joint score khi cùng biến/đơn vị, "
               "return score khi cùng return. Clustering không có log density được đánh giá riêng. Danh mục dùng tín hiệu cuối ngày t, "
               "khớp open t+2, thanh lý tại open ngày cuối; paired circular block bootstrap đánh giá bất định.</p>")
    parameters = pd.DataFrame([
        ("K HMM chính", "BTC 4; ETH 5", "Quy tắc validation + recovery hiện hành trong lưới 2–5; không ấn định số trạng thái thật."),
        ("EM / khởi tạo", "300 vòng; tol 0,01; 20 seeds đầu, 3 seeds refit", "Nhiều khởi tạo giảm phụ thuộc local optimum; đây là thiết kế tính toán."),
        ("Recovery", "100 mô phỏng; RMSE transition ≤0,05; σ tương đối ≤0,20; convergence ≥0,90", "Mức chính xác nội bộ chưa hiệu chuẩn bằng tổn thất kinh tế; không phải chuẩn khoa học phổ quát."),
        ("TVTP / MS0", "Hai trạng thái, cùng nhóm biến và mẫu", "Giữ cùng cấu trúc đối chứng; transition logit dùng covariates trễ, không khẳng định K=2 tối ưu."),
        ("MS-GARCH", "Hai parallel GARCH(1,1), common mean; refit tháng", "Mô hình independent variance; softmax bảo đảm alpha,beta>0 và alpha+beta<1, không là chọn số trạng thái bằng test."),
        ("HDBSCAN density", "min_cluster_size BTC17 / ETH23; min_samples=4", "Duyệt 2..floor(n_train/2), tối đa train relative_validity; tie chọn size lớn. min_samples=d+1 là heuristic, không chuẩn tối ưu."),
        ("BOCPD", "Hazard BTC0,024517 / ETH0,026254; prior_weight=1", "Hazard ứng viên = 1/duration HMM train; chọn bằng validation log score cùng prior_weight 1/n hoặc 1. Prior Normal-Inverse-Gamma alpha=2, mean/variance từ quá khứ."),
        ("Danh mục", "Mục tiêu σ 40%/năm; exposure 0–100%; tiền mặt 0%", "Kịch bản rủi ro cố định, không tối ưu bằng test; wRV=min(1;0,4/rv30), wR=wRV×(1−Pσcao)."),
        ("Chi phí", "5 / 10 / 25 bps mỗi chiều; chính 10", "Kịch bản độ nhạy, không phải biểu phí thực đo từ Binance."),
        ("Khoảng tin cậy", "95%; 1.000 bootstrap; block chính 30 ngày", "Giữ phụ thuộc thời gian; Sharpe có độ nhạy block 7/60. Không hiệu chỉnh nhiều so sánh."),
        ("Năm hóa", "365 ngày", "Thị trường crypto giao dịch mỗi ngày."),
    ], columns=["Parameter", "Value", "Basis / interpretation"])
    add("2. Phương pháp và tham số", methods + table(parameters))
    selection = store.frame("model_selection.csv")
    simulation = store.frame("simulation_summary.csv")
    simulation = simulation.loc[simulation.heavy_tail.eq(False)].copy()
    add("Lựa chọn HMM", table(selection[["asset", "k", "validation_log_score", "train_bic", "train_aic", "n_parameters"]])
        + table(simulation[["asset", "k", "n_days", "transition_rmse", "transition_rmse_mcse", "volatility_relative_rmse", "convergence_rate"]])
        + "<p>BTC K=5 có validation log score cao hơn K=4 nhưng không qua recovery hiện hành. RMSE trong bảng là trung bình RMSE "
          "của từng mô phỏng, không phải RMSE gộp. MCSE BTC K=4 ≈0,0102 cho thấy quyết định gần ngưỡng còn bất định; giữ nguyên theo yêu cầu.</p>")
    add("3. Trạng thái thị trường", table(profiles, {"mean_return_daily": ".3%", "volatility_annual": ".1%", "occupancy": ".1%"})
        + '<p>' + LABEL_NOTE + ' Nhãn là mô tả kinh tế [7–8], không phải nhãn thật được giám sát; mean emission ETH R3/R4 có CI chứa 0.</p>')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]})
    for col, asset in enumerate(ASSETS):
        value = store.frame(f"{asset}_HMM.parquet")
        price = store.frame(f"{asset}_market.parquet").loc[value.index, "close"]
        ax = axes[0, col]
        ax.plot(price.index, price, color="#6b7280", lw=.6)
        for state in range(4 if asset == "BTCUSDT" else 5):
            mask = value.state.eq(state)
            ax.scatter(price.index[mask], price[mask], s=5, color=COLORS[state], label=f"R{state+1}")
        ax.set(title=asset[:3] + " · HMM", ylabel="USDT")
        ax.legend(ncol=5, fontsize=8, loc="upper left")
        cols = [f"p_{i}" for i in range(4 if asset == "BTCUSDT" else 5)]
        axes[1, col].stackplot(value.index, value[cols].to_numpy().T, colors=COLORS[:len(cols)])
        axes[1, col].set(ylim=(0, 1), ylabel="Filtered probability")
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
    fig.autofmt_xdate()
    fig.tight_layout()
    sections.append(figure(fig, "Hình 2. Giá và xác suất HMM trong 638 ngày test; tên của R theo bảng hồ sơ đầu kỳ ở trên."))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for asset, ax in zip(ASSETS, axes, strict=True):
        matrix = np.asarray(store.json(f"{asset}_initial_states.json")["transition"])
        ax.imshow(matrix, vmin=0, vmax=1, cmap="GnBu")
        labels = [f"R{i+1}" for i in range(len(matrix))]
        ax.set(xticks=range(len(matrix)), xticklabels=labels, yticks=range(len(matrix)), yticklabels=labels,
               title=asset[:3], xlabel="Next day", ylabel="Current day")
        for (i, j), v in np.ndenumerate(matrix):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v>.5 else "#20324a")
    fig.tight_layout()
    sections.append(figure(fig, "Hình 3. Ma trận chuyển HMM đầu test, fit trên lịch sử đến 31.12.2024; mỗi hàng tổng bằng 1."))
    score_cols = ["asset", "model", "test_days", "mean_return_log_score"]
    add("4. Đánh giá ngoài mẫu", table(scores[score_cols].dropna(subset=["mean_return_log_score"]))
        + "<p>Mean log density càng cao càng tốt. So sánh joint density của HMM/GMM dưới đây dùng cùng ba biến lõi; "
          "joint score HMM_B/C có số chiều khác nên không xếp chung.</p>"
        + table(scores.loc[scores.score_group.eq("joint_core"), ["asset", "model", "mean_log_score"]]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for asset, ax in zip(ASSETS, axes, strict=True):
        rows = scores.loc[scores.asset.eq(asset)].dropna(subset=["mean_return_log_score"]).sort_values("mean_return_log_score")
        ax.barh(rows.model, rows.mean_return_log_score, color=COLORS[0])
        ax.set(title=asset[:3], xlabel="Mean return log score")
    fig.tight_layout()
    sections.append(figure(fig, "Hình 4. Toàn bộ mô hình có predictive return density (19/tài sản); không gán điểm density cho clustering."))
    comparisons = store.table("score_comparisons").copy()
    comparisons["left"] = comparisons.left.fillna(comparisons.model)
    comparisons["right"] = comparisons.right.fillna(comparisons.reference)
    comparisons["Chỉ tiêu"] = np.where(comparisons.left.eq("HMM") & comparisons.right.eq("GMM"), "Joint lõi", "Return")
    add("So sánh ghép cặp", table(comparisons[["asset", "Chỉ tiêu", "left", "right", "n_common", "mean_difference", "ci_low", "ci_high"]])
        + table(scores.loc[scores.model.str.startswith(("HDBSCAN", "KMeans")), ["asset", "model", "state_changes", "noise_fraction", "mean_membership_strength"]], {"noise_fraction": ".1%"})
        + "<p>Membership strength HDBSCAN là độ thuộc cụm, không phải xác suất thị trường căng thẳng. Các cấu hình coarse ra toàn noise vẫn được giữ.</p>")
    principal = performance.loc[performance.model.eq("HMM") & performance.cost_bps.eq(10)].copy()
    principal["strategy"] = principal.strategy.map(STRATEGIES)
    port_cols = ["asset", "model", "strategy", "cagr", "volatility", "sharpe", "max_drawdown", "average_exposure"]
    fmt = {c: ".2%" for c in ("cagr", "volatility", "max_drawdown", "average_exposure")}
    add("5. Hiệu quả danh mục", table(principal[port_cols], fmt))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
    for col, asset in enumerate(ASSETS):
        for i, (strategy, label) in enumerate(STRATEGIES.items()):
            path = store.frame(f"{asset}_HMM_{strategy}_portfolio.parquet")
            axes[0, col].plot(path.index, path.equity, label=label, color=COLORS[i])
            axes[1, col].plot(path.index, path.drawdown, color=COLORS[i])
        axes[0, col].set(title=asset[:3], ylabel="Portfolio value, initial = 1")
        axes[0, col].legend(fontsize=8)
        axes[1, col].set(ylabel="Drawdown")
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
    fig.autofmt_xdate()
    fig.tight_layout()
    sections.append(figure(fig, "Hình 5. Ba danh mục HMM chính sau phí 10 bps mỗi chiều; cùng lịch khớp lệnh và thanh lý cuối kỳ."))
    intervals = store.table("portfolio_uncertainty")
    add("Bất định hiệu quả", table(intervals.loc[intervals.block_days.eq(30)])
        + "<details><summary>Toàn bộ danh mục và chi phí 5/10/25 bps</summary>"
        + table(performance[["asset", "model", "strategy", "cost_bps", "cagr", "volatility", "sharpe", "max_drawdown", "average_exposure"]], fmt) + "</details>")
    add("6. Diễn giải và kết luận", "<p>HMM chính tách được các hồ sơ return–volatility khác nhau và có joint log score cao hơn GMM "
        "(BTC 2,1180 so với 1,1764; ETH 1,6836 so với 0,6824). Trong density return, chênh lệch MSGARCH với HMM_return chưa chắc chắn "
        "vì CI 95% chứa 0 ở cả hai tài sản. TVTP_MACRO BTC có cải thiện nhỏ so với MS0 cùng mẫu (+0,00475), nhưng kiểm tra trễ thêm 7 ngày "
        "có CI chứa 0; không suy ra quan hệ nhân quả. HDBSCAN density cho kết quả khác biệt mạnh giữa BTC và ETH. "
        "Sau phí 10 bps, CAGR HMM Regime-aware là −10,25% BTC và −12,31% ETH, thấp hơn Volatility targeting (−9,27%; −7,18%). "
        "Kết quả ủng hộ mô tả và giám sát trạng thái; chưa có bằng chứng bền vững rằng chiến lược hiện tại tạo hiệu quả vượt trội. "
        "Giữ quy tắc mẫu/ngưỡng hiện tại để thảo luận tiếp, không gọi chúng là các lựa chọn đã tối ưu khoa học.</p>")
    sources = store.table("data_sources").copy()
    sources["Chuỗi"] = sources.series.fillna(sources.dataset)
    sources["Nguồn"] = sources.source.fillna("Binance")
    sources["Quan sát"] = sources.observations.fillna(sources.valid_days)
    add("Bảng tra cứu dữ liệu và tham số", '<details><summary>Nguồn, nhóm biến, parameter confidence intervals, độ ổn định</summary>'
        + table(sources[["Chuỗi", "asset", "Nguồn", "first_date", "last_date", "Quan sát"]])
        + table(counts[["asset", "cohort", "split", "start", "end", "complete_feature_days", "features"]])
        + table(store.frame("parameter_intervals.csv"))
        + table(store.frame("learning_curve.csv"))
        + table(store.frame("extensions/liquidity_profiles.csv")) + '</details>')
    links = ''.join(f'<li><a href="{url}">{escape(label)}</a></li>' for label, url in SOURCES)
    digest = sha256(store.archive.read_bytes()).hexdigest()
    add("Tài liệu và tái lập", '<ol>' + links + '</ol><p>Nguồn số liệu: kho kết quả đã chốt trong repo; '
        'tạo lại bằng <code>python -m regime.cli report</code>, không tải dữ liệu hoặc fit lại. '
        'SHA256 research.zip: <code>' + digest + '</code>. File chứa toàn bộ hình SVG và bảng, mở offline.</p>')
    css = """body{font:15px/1.6 'Segoe UI',Arial,sans-serif;color:#243449;background:#f4f7fa;margin:0}
main{max-width:1160px;margin:auto;background:white;padding:40px}h1{font-size:30px;line-height:1.25}h2{font-size:21px;color:#147d78;margin-top:36px}
p{max-width:1050px}.meta,figcaption{color:#607085;font-size:13px}.table{overflow-x:auto;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:8px 9px;border-bottom:1px solid #e3eaf0;text-align:right}
th{background:#edf5f5;white-space:nowrap}td:first-child,td:nth-child(2){text-align:left}tbody tr:nth-child(even){background:#fafcfd}
figure{margin:24px 0}svg{width:100%;height:auto}details{padding:12px 0}summary{cursor:pointer;color:#147d78}a{color:#126d9c}code{overflow-wrap:anywhere}
@media(max-width:650px){main{padding:18px}h1{font-size:24px}}@media print{body{background:white}main{padding:0}figure{break-inside:avoid}}
"""
    html = ('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Kết quả nghiên cứu trạng thái thị trường crypto</title><style>' + css + '</style><main>'
            '<h1>Trạng thái thị trường BTC và ETH</h1><p class="meta">Tổng hợp kết quả · Snapshot 30.09.2026 · '
            '48 cấu hình / 2 tài sản · Test 01.01.2025–30.09.2026</p>' + ''.join(sections) + '</main></html>')
    output = ROOT / "results/tom_tat_ket_qua.html"
    output.write_text(html, encoding="utf-8", newline="\n")
    print(f"Report: {output} ({output.stat().st_size:,} bytes)")
    return output
