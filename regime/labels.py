"""Economic descriptions of the primary HMM initial-fit profiles only."""
STATE_NAMES = {
    "BTCUSDT": ("Thiên tăng, ít biến động", "Trung tính", "Thiên tăng, biến động cao", "Biến động rất cao"),
    "ETHUSDT": ("Thiên tăng, ít biến động", "Trung tính", "Thiên tăng, biến động cao",
                "Thiên giảm, biến động cao", "Biến động rất cao"),
}
LABEL_NOTE = ("Nhãn diễn giải theo hồ sơ HMM đến 31.12.2024; mã R được ghép qua các lần fit tháng. "
              "Mức biến động tương đối trong từng tài sản; thiên tăng/giảm mô tả hướng lợi suất trung bình.")


def state_label(asset, model, state):
    """Keep unrelated models' state identifiers separate from primary HMM labels."""
    state = int(state)
    if state < 0:
        return "Noise"
    code = f"R{state + 1}"
    names = STATE_NAMES.get(asset, ()) if model == "HMM" else ()
    return f"{code} · {names[state]}" if state < len(names) else code
