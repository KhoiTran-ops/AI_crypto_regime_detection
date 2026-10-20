"""Display-only regime names grounded in each model's saved state ordering."""
import re

import pandas as pd

STATE_NAMES = {
    "BTCUSDT": ("Bullish regime — low volatility", "Neutral market regime",
                "Bullish regime — high volatility", "High-volatility regime"),
    "ETHUSDT": ("Bullish regime — low volatility", "Neutral market regime", "Bullish regime — high volatility",
                "Bearish regime — high volatility", "High-volatility regime"),
}
LABEL_NOTE = ("Nhãn diễn giải theo hồ sơ HMM đến 31.12.2024; mã R được ghép qua các lần fit tháng. "
              "Volatility được so tương đối trong từng tài sản; Bullish/Bearish mô tả hướng mean return.")


def state_label(asset, model, state, k=None, high_state=None):
    """Preserve state IDs; attach economic names only when their ordering is known."""
    state = int(state)
    if state < 0:
        return "Noise · Unclassified observations"
    code = f"S{state + 1} raw" if model == "MSGARCH" else f"R{state + 1}"
    if model == "HMM":
        names = STATE_NAMES.get(asset, ())
        return f"{code} · {names[state]}" if state < len(names) else code
    if model == "MSGARCH":
        if high_state is None:
            return code
        name = "High-volatility regime" if state == high_state else "Low-volatility regime"
    elif k is not None:
        k = int(k)
        if not 0 <= state < k:
            raise ValueError("State ID outside the fitted state count")
        kind = "cluster" if model == "KMeans" or model.startswith("HDBSCAN") else "regime"
        if k == 1:
            name = f"Single market {kind}"
        elif state == 0:
            name = f"Low-volatility {kind}"
        elif state == k - 1:
            name = f"High-volatility {kind}"
        else:
            name = f"Intermediate-volatility {kind} (rank {state+1}/{k})"
    else:
        return code
    return f"{code} · {name}"


def label_note(model):
    if model == "HMM":
        return LABEL_NOTE
    if model == "BOCPD":
        return "BOCPD hiển thị change-point probability; không tự tạo regime ID hoặc ngưỡng phân loại."
    if model.startswith("HDBSCAN"):
        return ("Nhãn theo thứ bậc return volatility trên train của mỗi lần fit; số cluster có thể đổi theo tháng. "
                "Noise là quan sát chưa được gán cụm; membership strength không phải stress probability.")
    if model == "MSGARCH":
        return ("Nhãn Low/High theo unconditional variance của từng lần fit; S1/S2 là raw state ID. "
                "Biểu đồ probability được trình bày theo Low/High để tránh đảo ý nghĩa khi raw ID đổi.")
    if model.startswith("HMM"):
        return ("Nhãn theo thứ bậc return volatility của fit đầu kỳ; các refit được ghép state bằng Hungarian. "
                "Low/Intermediate/High là mức tương đối trong mô hình, không dùng ngưỡng volatility tuyệt đối.")
    return ("Nhãn theo thứ bậc return volatility/innovation volatility trên train của mỗi lần fit. "
            "Low/Intermediate/High là mức tương đối trong mô hình, không dùng ngưỡng volatility tuyệt đối.")


def label_predictions(asset, model, value, diagnostics):
    """Annotate a copy; all saved numerical predictions and state IDs remain untouched."""
    output = value.copy()
    if "state" not in output:
        return output
    k = sum(bool(re.fullmatch(r"p_\d+", c)) for c in value)
    if model == "MSGARCH" or model.startswith("HDBSCAN"):
        field = "high_unconditional_variance_state" if model == "MSGARCH" else "clusters"
        by_fit = pd.Series({pd.Timestamp(d["period"]): d[field] for d in diagnostics}).sort_index()
        context = by_fit.reindex(value.index, method="ffill")
        if context.isna().any():
            raise ValueError("Missing fit diagnostics for regime interpretation")
        if model == "MSGARCH":
            output["display_high_state"] = context.astype(int)
            output["market_regime"] = [state_label(asset, model, s, high_state=int(h))
                                       for s, h in zip(value.state, context, strict=True)]
        else:
            output["market_regime"] = [state_label(asset, model, s, k=int(n))
                                       for s, n in zip(value.state, context, strict=True)]
    else:
        output["market_regime"] = value.state.map(lambda s: state_label(asset, model, s, k=k or None))
    return output
