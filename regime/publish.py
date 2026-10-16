"""Publish tables, scientific figures and a Vietnamese manuscript from locked results."""
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT, config, save_json
from .research import dataset

RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
PALETTE = ["#3566A8", "#237E83", "#9F781F", "#B04D2C", "#A94456"]


def md_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else f"{value:.4f}"
        return str(value).replace("|", "/").replace("\n", " ")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def publish_data():
    cfg = config()
    quality, counts, descriptive = [], [], []
    for kind in ("spot", "funding", "oi"):
        audit = json.loads((ROOT / f"outputs/data_audit/{kind}.json").read_text())
        for item in audit:
            daily = pd.read_parquet(ROOT / f"data/processed/{item['asset']}_{kind}.parquet")
            field = {"spot": "close", "funding": "funding", "oi": "oi"}[kind]
            quality.append({"asset": item["asset"], "dataset": kind,
                "first_date": str(daily.index.min().date()), "last_date": str(daily.index.max().date()),
                "expected_days": len(daily), "valid_days": int(daily[field].notna().sum()),
                "coverage": daily[field].notna().mean(),
                "invalid_snapshots": item.get("invalid_snapshots", 0),
                "missing_days": int(daily[field].isna().sum())})
    for asset in cfg["assets"]:
        market, _ = dataset(asset)
        derivatives, _ = dataset(asset, "C")
        market = market.join(derivatives[["oi", "oi_quote", "oi_change"]]).join(
            pd.read_parquet(ROOT / f"data/processed/{asset}_funding.parquet"))
        market.to_parquet(RESULTS / f"{asset}_market.parquet")
        for cohort in ("A", "B", "C"):
            frame, columns = dataset(asset, cohort)
            train_end = "2023-12-31" if cohort == "C" else cfg["train_end"]
            validation_start = "2024-01-01" if cohort == "C" else cfg["validation_start"]
            for label, start, end in (("train", str(frame.index.min().date()), train_end),
                                     ("validation", validation_start, cfg["validation_end"]),
                                     ("test", cfg["test_start"], cfg["test_end"])):
                subset = frame.loc[start:end]
                clean = subset.dropna(subset=columns)
                counts.append({"asset": asset, "cohort": cohort, "split": label,
                    "start": start, "end": end, "calendar_days": len(subset),
                    "complete_feature_days": len(clean), "excluded_days": len(subset)-len(clean),
                    "features": ", ".join(columns)})
                if cohort == "A":
                    returns = clean["return"]
                    descriptive.append({"asset": asset, "split": label, "n": len(clean),
                        "mean_log_return": returns.mean(), "daily_sd": returns.std(ddof=1),
                        "annual_volatility": returns.std(ddof=1)*np.sqrt(365),
                        "skewness": returns.skew(), "excess_kurtosis": returns.kurt(),
                        "return_q05": returns.quantile(.05), "return_q95": returns.quantile(.95),
                        "median_quote_volume": clean.quote_volume.median(),
                        "median_illiquidity_proxy": clean.illiquidity.median()})
    import hashlib
    inventory = []
    for path in sorted((ROOT / "data/raw").rglob("*.zip")):
        checksum = path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Raw archive changed: {path.name}")
        key = path.relative_to(ROOT / "data/raw").as_posix()
        inventory.append({"file": path.relative_to(ROOT).as_posix(),
                          "url": "https://data.binance.vision/" + key,
                          "sha256": checksum, "bytes": path.stat().st_size})
    pd.DataFrame(inventory).to_csv(ROOT / "data/manifest_archives.csv", index=False)
    pilot = ROOT / "outputs/pilot_absolute_volume/adequacy/simulation_summary.csv"
    if pilot.exists():
        table = pd.read_csv(pilot)
        table["status"] = "Rejected pilot; absolute volume and single-start estimator"
        table.to_csv(RESULTS / "pilot_absolute_volume_summary.csv", index=False)
    pd.DataFrame(quality).to_csv(RESULTS / "data_quality.csv", index=False)
    pd.DataFrame(counts).to_csv(RESULTS / "sample_counts.csv", index=False)
    pd.DataFrame(descriptive).to_csv(RESULTS / "descriptive_statistics.csv", index=False)
    sources = ROOT / "outputs/research"
    for path in [sources / "model_selection.csv", sources / "benchmark_selection.csv",
                 sources / "cohort_selection.csv", *sorted((sources / "adequacy").glob("*.csv"))]:
        if "recovery" not in path.name and "bootstrap" not in path.name and "heavy_tail" not in path.name:
            shutil.copy2(path, RESULTS / path.name)
    simulation = pd.read_csv(RESULTS / "simulation_summary.csv")
    extras = []
    for row in simulation.itertuples():
        path = sources / "adequacy" / (
            f"{row.asset}_heavy_tail.csv" if row.heavy_tail else f"{row.asset}_k{row.k}_recovery.csv")
        records = pd.read_csv(path)
        extras.append({"transition_rmse_mcse": records.transition_rmse.std(ddof=1)/np.sqrt(records.converged.sum()),
                       "volatility_rmse_mcse": records.volatility_relative_rmse.std(ddof=1)/np.sqrt(records.converged.sum()),
                       "minimum_visited_states": records.visited_states.min()})
    pd.concat([simulation, pd.DataFrame(extras)], axis=1).to_csv(RESULTS / "simulation_summary.csv", index=False)


def figures():
    folder = PAPER / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    for asset in config()["assets"]:
        market = pd.read_parquet(RESULTS / f"{asset}_market.parquet")
        prediction = pd.read_parquet(RESULTS / f"{asset}_HMM.parquet")
        states = json.loads((RESULTS / f"{asset}_initial_states.json").read_text())
        probability_columns = [name for name in prediction if name.startswith("p_") and name != "p_high_vol"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1, 1]})
        dates = prediction.index
        axes[0].plot(dates, market.close.reindex(dates), color="#354258", lw=.8)
        for state in range(len(probability_columns)):
            points = prediction.index[prediction.state.eq(state)]
            axes[0].scatter(points, market.close.reindex(points), s=5, color=PALETTE[state], label=f"R{state+1}")
        axes[0].set_ylabel("Giá đóng cửa (USDT)")
        axes[0].legend(ncol=5, loc="upper left")
        axes[0].set_title(f"{asset}: giá, xác suất lọc và độ bất định trạng thái ngoài mẫu")
        axes[1].stackplot(dates, prediction[probability_columns].to_numpy().T,
                          colors=PALETTE[:len(probability_columns)], alpha=.85)
        axes[1].set_ylabel("Xác suất")
        axes[1].set_ylim(0, 1)
        axes[2].plot(dates, prediction.entropy, color="#3566A8")
        axes[2].set_ylabel("Entropy")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(folder / f"{asset}_regimes.{suffix}", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for name, label in (("buy_hold", "Mua và giữ"), ("vol_target", "Volatility targeting"),
                            ("regime_aware", "Regime-aware")):
            portfolio = pd.read_parquet(RESULTS / f"{asset}_HMM_{name}_portfolio.parquet")
            axes[0].plot(portfolio.index, portfolio.equity, label=label)
            axes[1].plot(portfolio.index, portfolio.drawdown*100)
        axes[0].set_title(f"{asset}: danh mục ngoài mẫu, chi phí 10 bps")
        axes[0].set_ylabel("Giá trị ban đầu = 1")
        axes[0].legend(ncol=3)
        axes[1].set_ylabel("Drawdown (%)")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(folder / f"{asset}_portfolio.{suffix}", bbox_inches="tight")
        plt.close(fig)

        transition = np.array(states["transition"])
        fig, axis = plt.subplots(figsize=(5.5, 4.5))
        im = axis.imshow(transition, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(transition)):
            for j in range(len(transition)):
                axis.text(j, i, f"{transition[i,j]:.3f}", ha="center", va="center",
                          color="white" if transition[i,j]>.6 else "#22324A")
        labels = [f"R{i+1}" for i in range(len(transition))]
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
                 yticklabels=labels, xlabel="Ngày kế tiếp", ylabel="Ngày hiện tại",
                 title=f"{asset}: transition, fit đến 31/12/2024")
        fig.colorbar(im, ax=axis)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(folder / f"{asset}_transition.{suffix}", bbox_inches="tight")
        plt.close(fig)


def manuscript():
    cfg = config()
    scores = pd.read_csv(RESULTS / "model_scores.csv")
    performance = pd.read_csv(RESULTS / "portfolio_performance.csv")
    core_performance = performance[performance.model.eq("HMM") & performance.cost_bps.eq(10)]
    uncertainty = pd.read_csv(RESULTS / "portfolio_uncertainty.csv")
    core_uncertainty = uncertainty[uncertainty.model.eq("HMM") & uncertainty.block_days.eq(30)]
    comparisons = pd.read_csv(RESULTS / "score_comparisons.csv")
    lock = json.loads((RESULTS / "protocol_lock.json").read_text())
    decisions = lock["protocol"]["core"]
    simulated = pd.read_csv(RESULTS / "simulation_summary.csv")
    paragraphs = []
    for asset in cfg["assets"]:
        perf = core_performance[core_performance.asset.eq(asset)].set_index("strategy")
        ci = core_uncertainty[core_uncertainty.asset.eq(asset)].iloc[0]
        delta = perf.loc["regime_aware","sharpe"] - perf.loc["vol_target","sharpe"]
        supported = ci.sharpe_difference_ci_low > 0
        paragraphs.append(
            f"**{asset}**: regime-aware có CAGR {perf.loc['regime_aware','cagr']:.2%}, "
            f"Sharpe {perf.loc['regime_aware','sharpe']:.3f} và drawdown tối đa "
            f"{perf.loc['regime_aware','max_drawdown']:.2%}; volatility targeting có Sharpe "
            f"{perf.loc['vol_target','sharpe']:.3f}. Chênh lệch Sharpe là {delta:+.3f}, "
            f"CI bootstrap 95% [{ci.sharpe_difference_ci_low:.3f}; {ci.sharpe_difference_ci_high:.3f}]. "
            + ("Khoảng này hỗ trợ cải thiện trong thí nghiệm cụ thể, với giới hạn của bootstrap và mẫu."
               if supported else "Khoảng này không hỗ trợ kết luận cải thiện Sharpe một cách rõ ràng."))
    findings = "\n\n".join(paragraphs)
    reference = core_performance.query("asset == 'ETHUSDT' and strategy == 'vol_target'").iloc[0]
    alternative = performance.query("asset == 'ETHUSDT' and model == 'NHHMM' and strategy == 'regime_aware' and cost_bps == 10").iloc[0]
    intervals30 = uncertainty[uncertainty.block_days.eq(30)]
    positive = int(intervals30.sharpe_difference_ci_low.gt(0).sum())
    comparison_note = (f"Các đối chứng có kết quả điểm khác mô hình chính; ví dụ ETH NHHMM có CAGR "
        f"{alternative.cagr:.2%}, so với {reference.cagr:.2%} của volatility targeting. "
        f"Trong toàn bộ {len(intervals30)} phép so sánh tại 10 bps, có {positive} khoảng tin cậy chênh lệch "
        "Sharpe với cận dưới dương. Không chọn lại mô hình chính theo xếp hạng test.")

    abstract_findings = []
    for asset in cfg["assets"]:
        perf = core_performance[core_performance.asset.eq(asset)].set_index("strategy")
        abstract_findings.append(f"{asset}: CAGR regime-aware {perf.loc['regime_aware','cagr']:.2%}, "
                                 f"volatility targeting {perf.loc['vol_target','cagr']:.2%}")
    abstract_findings = "Kết quả tại chi phí 10 bps: " + "; ".join(abstract_findings) + "."

    profiles = []
    for asset in cfg["assets"]:
        items = json.loads((RESULTS / f"{asset}_initial_states.json").read_text())["states"]
        for item in items:
            profiles.append({"asset": asset, "regime": f"R{item['state']+1}",
                "mean_return_daily": item["mean_return_daily"], "volatility_annual": item["volatility_annual"],
                "occupancy": item["occupancy"], "persistence": item["persistence"],
                "expected_duration_days": item["expected_duration_days"]})
    profiles = pd.DataFrame(profiles)
    profiles.to_csv(RESULTS / "regime_profiles.csv", index=False)
    driver_rows = []
    for asset in cfg["assets"]:
        coefficients = pd.read_csv(RESULTS / f"{asset}_NHHMM.coefficients.csv")
        diagnostics = json.loads((RESULTS / f"{asset}_NHHMM.diagnostics.json").read_text())
        period = coefficients.period.min()
        subset = coefficients[coefficients.period.eq(period) & coefficients.parameter.str.contains("tvtp")]
        for row in subset.to_dict("records"):
            driver_rows.append({"asset": asset, "raw_high_vol": diagnostics[0]["raw_high_vol_state"],
                                **{name: row[name] for name in ("parameter","estimate","std_error","ci_low","ci_high")}})
    driver_table = pd.DataFrame(driver_rows)
    driver_table.to_csv(RESULTS / "transition_drivers_initial.csv", index=False)
    intervals = pd.read_csv(RESULTS / "parameter_intervals.csv")
    interval_rows = []
    for (asset, state), subset in intervals.groupby(["asset","state"]):
        values = subset.set_index("parameter")
        interval_rows.append({"asset": asset, "regime": f"R{state+1}",
            "volatility_low": values.loc["volatility","ci_low"], "volatility_high": values.loc["volatility","ci_high"],
            "duration_low": values.loc["duration","ci_low"], "duration_high": values.loc["duration","ci_high"]})
    interval_table = pd.DataFrame(interval_rows)
    provenance = ""
    if (RESULTS / "revision.json").exists():
        revision = json.loads((RESULTS / "revision.json").read_text(encoding="utf-8"))
        provenance = ("**Lịch sử phiên bản:** thiết kế ban đầu khóa trước lượt đánh giá đầu, hash "
            + revision["original_protocol_hash"] + ". Bản hiện tại sửa seed của Generator trong statsmodels "
            "sau khi kiểm tra tái lập phát hiện sai khác. Holdout đã được xem ở v1; đây là tái tính có công khai sửa lỗi, "
            "không phải một holdout mới chưa từng được xem. Không thay đặc tả, ngưỡng hoặc quy tắc theo điểm test.")
    text = f"""# Phát hiện trạng thái rủi ro của Bitcoin và Ethereum trên Binance
## Hidden Markov Models, Markov Switching và giá trị của tín hiệu trong quản trị danh mục

Bản thảo nghiên cứu tính từ dữ liệu thực. Ngày lập: 16/10/2026. Snapshot chốt: 30/09/2026 UTC.
Phiên bản protocol: {lock['protocol_hash']}.

{provenance}

## Tóm tắt

Nghiên cứu xây dựng mô hình trạng thái ẩn cho BTCUSDT và ETHUSDT từ lịch sử Binance bắt đầu tháng 8/2017. Bộ biến chính gồm log return, volatility thực hiện 30 ngày và khối lượng tương đối, với chuẩn hóa chỉ dùng quá khứ. Dữ liệu được kiểm tra checksum, lịch UTC, đơn vị timestamp, ngày thiếu và điều kiện quan sát đầy đủ. Thiết kế phân tách train đến 2022, validation 2023–2024 và đánh giá cuối 638 ngày từ 2025 đến 30/09/2026; đồng thời kiểm tra nhiều origin trước test. Trong lưới 2–5 trạng thái, quy tắc kết hợp điểm validation và recovery simulation chọn BTC {decisions['BTCUSDT']['k']} trạng thái, ETH {decisions['ETHUSDT']['k']} trạng thái. BTC 5 trạng thái bị loại dù có điểm validation cao hơn vì không đạt ngưỡng sai số transition. HMM được so sánh với GMM, K-means và Markov Switching AR; funding/OI được xem xét trên các cohort riêng. Hiệu quả kinh tế được đo bằng danh mục long/cash với thực thi trễ hai ngày và ba kịch bản chi phí. {abstract_findings} Kết quả phải được đọc cùng khoảng tin cậy, độ nhạy theo seed và giới hạn của mô hình Gaussian; phân tách trạng thái không tự chứng minh khả năng tạo lợi nhuận vượt trội.

**Từ khóa:** cryptocurrency; Hidden Markov Model; Markov Switching; time-series validation; quản trị rủi ro.

## 1. Đặt vấn đề và câu hỏi nghiên cứu

Biến động, hoạt động giao dịch và điều kiện phái sinh của crypto thay đổi theo thời gian. Một bộ phân loại cứng dựa trên mức giảm giá đơn lẻ không biểu diễn được bất định của trạng thái. HMM cho phép mô hình hóa phân phối quan sát phụ thuộc trạng thái ẩn và độ bền của từng trạng thái.

Nghiên cứu hỏi: (RQ1) số trạng thái nào trong lưới đặt trước vừa phù hợp ngoài mẫu vừa có khả năng khôi phục tham số; (RQ2) các trạng thái khác nhau về return và volatility ra sao; (RQ3) funding và volume trễ có liên quan đến transition hay bổ sung thông tin dự báo; (RQ4) HMM có biểu hiện khác GMM/K-means và Markov Switching không; (RQ5) giảm exposure theo xác suất trạng thái sigma cao nhất có cải thiện volatility targeting sau chi phí hay không.

Phạm vi là hai cặp spot trên một sàn và các biến perpetual Binance liên quan. Nghiên cứu không dùng VIX/DXY hoặc suy rộng tự động sang toàn thị trường. Các nhãn R1…RK là nhãn mô hình, không phải nhãn bull/bear/stress có ground truth.

## 2. Cơ sở phương pháp

Giudici và Abu Hashish (2020) nghiên cứu trạng thái ẩn của giá Bitcoin bằng HMM. Figà-Talamanca, Focardi và Patacca (2021) xem xét trạng thái riêng và chung của nhiều cryptocurrency; kết quả trong mẫu của họ không tạo ra một K cố định cho mọi thị trường. Koki, Leonardos và Piliouras (2022) so sánh các HMM Bayesian và NHHMM có predictors. Các nghiên cứu này tạo nền tảng cho câu hỏi về regime và dự báo, nhưng nghiên cứu hiện tại không nhận là tái tạo nguyên vẹn thiết kế hoặc kết quả của họ.

Pakštaitė et al. (2025) nghiên cứu HMM và các biến macro/Bitcoin; Shakourloo và Azimli (2026) xét MS-GARCH và Hidden Markov Copula. Hai hướng này giúp định vị phạm vi nhưng không được triển khai đầy đủ ở đây vì giới hạn Binance-only và mục tiêu codebase gọn. Đóng góp của dự án là quy trình audit dữ liệu, kiểm chứng khả năng ước lượng và so sánh ngoài mẫu minh bạch trên snapshot mới, không phải đề xuất một bộ ước lượng HMM mới. Literature matrix ghi rõ phần tài liệu chỉ được đọc abstract/preview, không trình bày đây là tổng quan hệ thống toàn văn.

Hamilton (1989) cung cấp nền tảng Markov Switching cho chuỗi thời gian có phân phối thay đổi theo trạng thái. Trong HMM, likelihood tốt và số trạng thái có ý nghĩa kinh tế là hai vấn đề khác nhau. Pohle et al. (2017) chỉ ra khó khăn khi chọn K chỉ bằng tiêu chí thông tin trong mô hình có sai đặc tả. Vì vậy nghiên cứu sử dụng cả validation, diagnostics và mô phỏng recovery.

Đánh giá giữ thứ tự thời gian và nhiều origin giúp quan sát sự phụ thuộc kết quả vào giai đoạn. Cerqueira, Torgo và Mozetič (2020) nghiên cứu các phương pháp ước lượng hiệu năng dự báo chuỗi thời gian. Thiết kế mô phỏng ở đây trình bày rõ mục tiêu, DGP, phương pháp và chỉ tiêu theo tinh thần hướng dẫn Morris, White và Crowther (2019). Những tài liệu này hỗ trợ cách làm; chúng không chứng minh sẵn rằng các mốc cuối 2022/2024 hoặc 1.902 ngày là tối ưu.

## 3. Dữ liệu, phạm vi mẫu và tiền xử lý

### 3.1. Nguồn và ngày bắt đầu

Spot lấy từ Binance public archive theo tháng và xác minh SHA-256. Archive 1 phút ngày đầu của cả hai cặp bắt đầu 17/08/2017 lúc 04:00 UTC, chỉ có 1.200 phút. Vì vậy raw giữ ngày 17/08 nhưng feature bắt đầu từ ngày đầy đủ 18/08. Sau 60 log return, ngày analysis đầu tiên là 17/10/2017. Cutoff 30/09/2026 là tháng hoàn chỉnh trước ngày thiết kế, phục vụ tái lập snapshot; không phải điểm gãy kinh tế.

Mỗi tài sản có 3.332 ngày spot raw, không thiếu ngày trong phạm vi. Timestamp spot chuyển từ milliseconds sang microseconds từ đầu 2025 theo tài liệu Binance và được chuẩn hóa trước khi ghép. Funding giữ từng settlement thực có, sau đó cộng theo ngày UTC. OI dùng snapshot dương cuối ngày trong 15 phút trước nửa đêm; ngày không đạt giữ thiếu. Có 1.763/1.765 ngày OI cuối ngày hợp lệ cho mỗi tài sản; các snapshot lỗi được đếm riêng và raw không sửa.

{md_table(pd.read_csv(RESULTS/'data_quality.csv'))}

### 3.2. Chia mẫu có điều kiện và các cohort

Mẫu A có 1.902 ngày train, 731 ngày validation và 638 ngày test. Ranh giới lịch được khai báo trước và được đối chiếu bằng recovery, learning curve và nested temporal origins 2020–2024. Không chọn một origin vì nó cho kết quả đẹp nhất.

Mẫu B chỉ bắt đầu 28/11/2019, sau ngày funding ETH đầu tiên chưa đầy đủ. Mẫu C dùng log-change OI từ 02/12/2021; train đến 2023, validation 2024. Số ngày hợp lệ khác lịch vì không lấp OI thiếu. Các mở rộng là thăm dò; chưa được chứng nhận recovery như A. Khi đánh giá đóng góp OI/funding, mật độ return được so trên cùng các ngày của holdout, không so trực tiếp joint density khác chiều.

{md_table(pd.read_csv(RESULTS/'sample_counts.csv')[['asset','cohort','split','calendar_days','complete_feature_days','excluded_days']])}

### 3.3. Biến và xử lý nhân quả

Return là ln(C_t/C_t−1). Volatility năm hóa bằng SD rolling return × sqrt(365), ddof=1. Volume tương đối là z-score log(1+quote volume) hôm nay so với 30 ngày trước; scaler mô hình chỉ fit lịch sử train. Không forward-fill giá/OI hoặc coi thiếu là 0; không nối return bắc qua ngày thiếu. Funding và OI được giữ riêng với spot vì khác thị trường.

Pilot khối lượng tuyệt đối tạo trạng thái gần hấp thụ, phản ánh tăng trưởng quy mô sàn nhiều hơn điều kiện tương đối. Thay đổi sang volume_z được quyết định bằng train trước test và lưu trong ADR-002. Kết quả pilot thất bại được giữ, không trình bày như đặc tả chính thành công.

Thống kê mô tả theo split:

{md_table(pd.read_csv(RESULTS/'descriptive_statistics.csv')[['asset','split','n','mean_log_return','annual_volatility','skewness','excess_kurtosis']])}

## 4. Mô hình, lựa chọn và thiết kế đánh giá

### 4.1. HMM và nhận diện trạng thái

Cho S_t thuộc {{1,…,K}}, P(S_t=j|S_t−1=i)=p_ij. Với vector Y_t gồm ba biến chính, giả định Y_t|S_t=k có phân phối Gaussian với mean riêng và covariance diagonal. Số tham số tự do là (K−1)+K(K−1)+2Kd; d=3 cho 15/26/39/54 tham số khi K=2/3/4/5.

Ước lượng EM dùng 20 khởi tạo đầu kỳ; chọn nghiệm hội tụ có likelihood train cao nhất. Mỗi đầu tháng refit expanding với ba khởi tạo cố định, chỉ dùng các ngày trước tháng đó. Chuỗi bị gián đoạn được phân đoạn, không nén thời gian. Nhãn đầu kỳ xếp theo sigma return; nhãn giữa các refit ghép theo profile bằng thuật toán Hungarian.

Tín hiệu dùng xác suất lọc P(S_t|Y_1:t), không dùng smoothing toàn mẫu. Xác suất ngày kế tiếp tính từ filtered probability nhân ma trận transition của mô hình đã fit. Mật độ return dự báo được tính từ prior trước quan sát ngày t. Mật độ joint có điều chỉnh Jacobian scaler.

### 4.2. Cơ sở về độ đủ mẫu

Ngưỡng dự án khai báo: trung bình RMSE toàn ma trận transition ≤0,05; RMSE tương đối sigma return ≤20%; tỷ lệ hội tụ ≥90%. Đây là lựa chọn độ chính xác, không phải chuẩn học thuật cố định. Mỗi ứng viên có 100 mẫu Gaussian DGP từ train, fit lại bằng cùng bộ ước lượng 20 khởi tạo. Sai số Monte Carlo của các trung bình được công bố.

{md_table(simulated[['asset','k','heavy_tail','repetitions','transition_rmse','transition_rmse_mcse','volatility_relative_rmse','passes_project_precision_targets']])}

BTC K=5 không qua ngưỡng; BTC K=4 và ETH K=5 qua. ETH chạm biên trên lưới nên không được gọi là số trạng thái thật tối ưu. Heavy-tail t5 là kiểm tra sai đặc tả riêng. Parametric bootstrap cho persistence, duration, mean và sigma được báo trong parameter_intervals.csv, chỉ dùng dữ liệu trước 2025; khoảng này có tính điều kiện theo Gaussian DGP và chỉ 100 lần lặp nên lượng hóa đuôi còn thô.

Learning curve 730/1.095/1.460/1.902 ngày và các seed cho thấy nhãn không hoàn toàn ổn định. K của outer folds cũng thay đổi theo lịch sử. Sai số Monte Carlo của RMSE transition BTC K=4 khoảng 0,0102, còn ETH K=5 khoảng 0,0085; quyết định đạt dựa trên ước lượng điểm gần ngưỡng cần được xem thận trọng, chưa phải chứng minh cỡ mẫu tối thiểu. Vì vậy việc qua recovery không đồng nghĩa mọi nghiệm hoặc mọi giai đoạn đều ổn định; không có kết luận rằng đúng ngày cuối 2022 là ranh giới duy nhất phù hợp.

### 4.3. Đối chứng và mở rộng

GMM diagonal chọn K bằng validation, K-means dùng cùng K với HMM chính. HMM return-only và Markov Switching AR(1) chọn K=2 hoặc 3 bằng validation. K-means chỉ có nhãn và ứng dụng danh mục, không có predictive density.

NHHMM mở rộng dùng Gaussian Markov Switching hai trạng thái, transition logit theo funding và volume_z trễ hai ngày. Mô hình MS0 transition cố định dùng đúng cùng cohort và ngày để đối chứng. Hai covariate chuẩn hóa bằng train. OI bổ sung dưới dạng emission trong HMM C; không tuyên bố đã ước lượng OI-TVTP khi dữ liệu bị thiếu.

### 4.4. Danh mục và suy luận

Volatility targeting dùng w=min(1,0,4/rv30); regime-aware nhân thêm (1−P_high_vol). Long/cash, không leverage/short, cash return 0. Tín hiệu close t giao dịch tại open t+2. Hai ngày đầu giữ cash, cuối kỳ thanh lý tại open 30/09/2026; không dùng giá tương lai. Hạch toán cập nhật giá trị tài sản trước tái cân bằng và tính phí trên notional giao dịch thật, gồm vào/ra.

Chi phí giả định 5/10/25 bps mỗi chiều; 10 bps là kết quả chính. Đây là kịch bản tổng chi phí giản lược, không phải báo giá phí Binance hoặc mô hình market impact. CAGR dùng số khoảng ngày; Sharpe dùng return tài khoản thực, năm hóa 365.

Chênh lệch mật độ và Sharpe dùng paired circular block bootstrap 1.000 lần, block chính 30 ngày; Sharpe kiểm tra thêm block 7/60. Các khoảng là xấp xỉ, không hiệu chỉnh nhiều so sánh. Thiết kế ban đầu được khóa hash trước lượt test đầu; bản sửa RNG và tái tính sau audit được công khai trong lịch sử phiên bản.

## 5. Kết quả

### 5.1. Lựa chọn và đặc trưng trạng thái

{md_table(pd.read_csv(RESULTS/'model_selection.csv')[['asset','k','validation_log_score','train_bic','n_parameters','min_posterior_days','limited_states']])}

Profile sau đây lấy từ mô hình đầu test, fit qua 31/12/2024; không coi là nhãn thật. Mean return và volatility là thống kê có trọng số posterior trên train. Persistence và duration là tham số mô hình, có thể rất bất định khi p gần 1.

{md_table(profiles)}

![BTC: trạng thái và xác suất ngoài mẫu](figures/BTCUSDT_regimes.png)

![ETH: trạng thái và xác suất ngoài mẫu](figures/ETHUSDT_regimes.png)

Các khoảng percentile bootstrap 95% dưới đây có điều kiện theo Gaussian HMM, fit qua 31/12/2024. SD là tham số emission năm hóa, khác thống kê SD có trọng số posterior của bảng profile; duration tính theo ngày. Khoảng này không bảo đảm coverage khi sai đặc tả:

{md_table(interval_table)}

![BTC: ma trận chuyển trạng thái trước test](figures/BTCUSDT_transition.png)

![ETH: ma trận chuyển trạng thái trước test](figures/ETHUSDT_transition.png)

Không ép mean dương/âm thành nhãn bull/bear khi chưa có kiểm chứng độc lập.

### 5.2. Mật độ dự báo và so sánh phương pháp

{md_table(scores[['asset','model','k','score_group','test_days','mean_log_score','mean_return_log_score','nonconverged_refits','state_changes']])}

{md_table(comparisons[['asset','left','right','n_common','mean_difference','ci_low','ci_high']])}

Joint core chỉ so HMM với GMM; return-only A chỉ so HMM return với MSAR; return-only B chỉ so NHHMM với MS0. HMM B/C so với HMM A bằng marginal return predictive density trên cùng ngày. Các phép so chéo cohort còn khác train history nên không thể quy mọi chênh lệch riêng cho thêm biến.

Số lần đổi nhãn phản ánh mức persistence của phương pháp nhưng ít đổi hơn không tự chứng minh đúng hơn. ARI giữa các phương pháp nằm trong method_comparison.csv, mang tính mô tả vì không có nhãn trạng thái ground truth. Cấu trúc rv30 rolling làm HMM có thể tận dụng độ bền của biến quan sát; ưu thế joint score vì thế không đồng nghĩa dự báo return vượt trội.

### 5.3. Hiệu quả kinh tế

{md_table(core_performance[['asset','strategy','cagr','volatility','sharpe','max_drawdown','turnover','ending_value']])}

{findings}

{comparison_note}

![BTC: danh mục và drawdown](figures/BTCUSDT_portfolio.png)

![ETH: danh mục và drawdown](figures/ETHUSDT_portfolio.png)

Toàn bộ ba mức chi phí và mọi mô hình được công bố trong portfolio_performance.csv. Không chọn lại target volatility hoặc ngưỡng xác suất theo kết quả test. Khi xác suất sigma cao nhất gần 0 trong phần lớn test, hai chiến lược gần nhau; giai đoạn như vậy không đủ xác thực năng lực phát hiện một cuộc khủng hoảng hiếm chưa xảy ra.

### 5.4. Funding, OI và transition drivers

Bảng hệ số từng tháng và CI tiệm cận nằm trong BTCUSDT_NHHMM.coefficients.csv và ETHUSDT_NHHMM.coefficients.csv. Tham số p[i→0].tvtp1 là hệ số funding chuẩn hóa, tvtp2 là volume chuẩn hóa; cần đối chiếu raw_high_vol_state của cùng tháng trước khi diễn giải chiều chuyển stress. Không gộp hệ số sau đảo nhãn như cùng một transition kinh tế.

Bảng sau là transition drivers của mô hình đầu test, ước lượng bằng dữ liệu trước 2025; các refit tháng khác có trong CSV. Raw state 0 là đích log odds so với raw state 1; cột raw_high_vol cho biết bên nào có sigma lớn hơn. Không đọc dấu hệ số như tác động cùng chiều đến stress khi chưa đối chiếu đích.

{md_table(driver_table)}

Cải thiện density của NHHMM nếu có là bằng chứng dự báo trong thiết kế này, không phải tác động nhân quả của funding. HMM C là khám phá trên lịch sử OI ngắn hơn; không được so BIC hoặc joint score khác chiều để tuyên bố OI tốt hơn. Các refit chưa hội tụ, nếu có, được công khai ở model_scores và diagnostics thay vì im lặng xóa khỏi mẫu.

## 6. Thảo luận, độ bền và giới hạn

Thứ nhất, ngày bắt đầu xuất phát từ nguồn dữ liệu thực và yêu cầu warm-up. Ngược lại, ranh giới train/validation là quyết định thiết kế được kiểm tra định lượng, không phải kết quả tất yếu của một chu kỳ Bitcoin. Nested origins 2020–2024 cho biết mô hình nhạy theo thời kỳ; không suy ngược lựa chọn K từ 2024 để tự nhận là dự báo hợp lệ cho 2020.

Thứ hai, recovery có điều kiện theo DGP vừa fit không chứng minh mô hình sinh đúng dữ liệu thị trường. Phân phối đuôi dày, trạng thái hiếm, phụ thuộc của rolling volatility và covariance diagonal có thể làm sai bất định. Seed sensitivity cho thấy nhiều cực trị địa phương; monthly refit ba seeds còn rủi ro hơn bộ kiểm chứng 20 seeds.

Thứ ba, dữ liệu là archive tải hồi cứu. Checksum và raw bất biến tái lập snapshot hiện có, không cung cấp đầy đủ historical vintages. Độ trễ hai ngày là giả định thực thi bảo thủ, không phải xác minh thời điểm mọi archive từng xuất hiện. Backtest không mô phỏng order book, trượt giá tùy thanh khoản, market impact, thuế hoặc lợi suất cash.

Thứ tư, chỉ hai cặp USDT trên một sàn có selection và venue bias. Funding/OI perpetual khác spot; thông tin phái sinh ngắn hơn không tạo được lịch sử từ 2017. Việc rút gọn macro là chủ ý theo phạm vi dữ liệu đã chọn.

Thứ năm, không có ground truth regimes. Khác biệt volatility là một phần cấu trúc mô hình; không dùng t-test sau phân loại để tự chứng minh các trạng thái “có thật”. Cần kiểm chứng độc lập, nhiều tài sản/sàn và prospective data để nâng mức khẳng định. Khoảng tin cậy có nhiều phép so sánh nên bằng chứng yếu cần được mô tả thận trọng.

## 7. Kết luận

Nghiên cứu bàn giao một quy trình tái lập từ Binance archive đến mô hình ngoài mẫu, bảng bất định và dashboard. Quy tắc validation kèm recovery chọn BTC bốn trạng thái, ETH năm trạng thái trong lưới đã đăng ký. Kết quả không cung cấp căn cứ để áp chung số trạng thái hoặc ngưỡng giao dịch cho mọi giai đoạn. Giá trị của mô hình cần được đánh giá đồng thời bằng mật độ dự báo, ổn định và hiệu quả danh mục sau chi phí; bằng chứng về một chiều không thay thế cho các chiều còn lại.

Dashboard biểu diễn xác suất lọc, xác suất ngày tiếp theo, lịch sử, transition, drivers, danh mục và chất lượng dữ liệu. Đây là công cụ kiểm tra kết quả và minh họa rủi ro theo mô hình, không phải hệ thống giao dịch đã được kiểm chứng prospective.

## Tài liệu tham khảo

1. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384. https://doi.org/10.2307/1912559
2. Pohle, J., Langrock, R., van Beest, F. M., & Schmidt, N. M. (2017). Selecting the number of states in hidden Markov models: pragmatic solutions illustrated using animal movement. Bản tác giả: https://arxiv.org/abs/1701.08673
3. Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: an empirical study on performance estimation methods. Bản tác giả: https://arxiv.org/abs/1905.11744
4. Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. Statistics in Medicine. https://doi.org/10.1002/sim.8086
5. Binance. Public data documentation. https://github.com/binance/binance-public-data
6. Binance. USD-M Futures market data API. https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data
7. hmmlearn. Tutorial and API, version 0.3.3. https://hmmlearn.readthedocs.io/en/stable/tutorial.html
8. statsmodels. MarkovAutoregression documentation. https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html

9. Giudici, P., & Abu Hashish, I. (2020). A hidden Markov model to detect regime changes in cryptoasset markets. https://doi.org/10.1002/qre.2673
10. Figà-Talamanca, G., Focardi, S., & Patacca, M. (2021). Regime switches and commonalities of the cryptocurrencies asset class. https://doi.org/10.1016/j.najef.2021.101425
11. Koki, C., Leonardos, S., & Piliouras, G. (2022). Exploring the predictability of cryptocurrencies via Bayesian hidden Markov models. https://doi.org/10.1016/j.ribaf.2021.101554
12. Pakštaitė, V., Filatovas, E., Juodis, M., & Paulavičius, R. (2025). Bitcoin Price Regime Shifts: A Bayesian MCMC and Hidden Markov Model Analysis of Macroeconomic Influence. https://doi.org/10.3390/math13101577
13. Shakourloo, A., & Azimli, A. (2026). Regime-switching in bitcoin volatility under global uncertainty: Markov-switching GARCH and hidden Markov Copula approaches. https://doi.org/10.1016/j.ribaf.2026.103295

## Phụ lục tái lập và tệp kết quả

- results/protocol_lock.json: lựa chọn, config, hash dữ liệu và code của phiên bản; lịch sử khóa ban đầu ở docs/history/.
- results/data_quality.csv, sample_counts.csv, descriptive_statistics.csv: audit và mô tả.
- results/model_selection.csv, simulation_summary.csv, learning_curve.csv, *_nested_origins.csv: cơ sở chọn mẫu/K.
- results/regime_profiles.csv, parameter_intervals.csv, *_initial_states.json: đặc trưng và bất định.
- results/model_scores.csv, score_comparisons.csv, method_comparison.csv: kết quả mô hình.
- results/portfolio_performance.csv, portfolio_uncertainty.csv: kinh tế và bất định.
- paper/figures/: PNG và PDF có thể đưa vào bài.
- docs/DATA_DICTIONARY.md, METHODOLOGY.md và decisions/: định nghĩa và lý do quyết định.
- README.md: môi trường, lệnh tái lập và cách mở dashboard.

Thực thi và dữ liệu chốt theo protocol hash ở đầu bản thảo. Các số liệu đều đọc từ artifact kết quả; không có dữ liệu minh họa giả lập được dùng thay cho kết quả thực. Mô phỏng chỉ phục vụ kiểm chứng phương pháp và được ghi rõ là mô phỏng.
"""
    PAPER.mkdir(exist_ok=True)
    (PAPER / "bao_cao_nghien_cuu.md").write_text(text, encoding="utf-8")


def publish():
    if not (RESULTS / "run.json").exists():
        raise ValueError("Holdout evaluation must complete before publishing")
    publish_data()
    figures()
    manuscript()
    update_hashes()


def update_hashes():
    artifacts = {}
    for folder in (RESULTS, PAPER):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "artifact_hashes.json":
                import hashlib
                artifacts[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(RESULTS / "artifact_hashes.json", artifacts)
    print(f"Published {len(artifacts)} research artifacts", flush=True)
