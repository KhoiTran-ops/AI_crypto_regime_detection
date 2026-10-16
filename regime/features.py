"""Causal daily features; missing dates invalidate rolling windows."""
import numpy as np
import pandas as pd

CORE_FEATURES = ["return", "rv30", "volume_z"]


def build_features(spot):
    index = pd.date_range(spot.index.min(), spot.index.max(), tz="UTC", name="date")
    frame = spot.reindex(index).copy()
    frame["return"] = np.log(frame.close / frame.close.shift(1))
    for window in (7, 30, 60):
        frame[f"rv{window}"] = frame["return"].rolling(window).std(ddof=1) * np.sqrt(365)
    frame["log_volume"] = np.log1p(frame.quote_volume)
    prior = frame.log_volume.shift(1).rolling(30)
    frame["volume_z"] = (frame.log_volume - prior.mean()) / prior.std(ddof=1).replace(0, np.nan)
    frame["illiquidity"] = frame["return"].abs() / frame.quote_volume.replace(0, np.nan)
    if {"high", "low"} <= set(frame):
        frame["range"] = np.log(frame.high / frame.low)
    return frame.replace([np.inf, -np.inf], np.nan)
