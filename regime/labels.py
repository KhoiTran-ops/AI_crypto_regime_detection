"""Economic descriptions of the primary HMM initial-fit profiles only."""
STATE_NAMES = {
    "BTCUSDT": ("Bullish regime — low volatility", "Neutral market regime",
                "Bullish regime — high volatility", "High-volatility regime"),
    "ETHUSDT": ("Bullish regime — low volatility", "Neutral market regime", "Bullish regime — high volatility",
                "Bearish regime — high volatility", "High-volatility regime"),
}
LABEL_NOTE = ("Nhãn diễn giải theo hồ sơ HMM đến 31.12.2024; mã R được ghép qua các lần fit tháng. "
              "Volatility được so tương đối trong từng tài sản; Bullish/Bearish mô tả hướng mean return.")


def state_label(asset, model, state):
    """Keep unrelated models' state identifiers separate from primary HMM labels."""
    state = int(state)
    if state < 0:
        return "Noise"
    code = f"R{state + 1}"
    names = STATE_NAMES.get(asset, ()) if model == "HMM" else ()
    return f"{code} · {names[state]}" if state < len(names) else code
