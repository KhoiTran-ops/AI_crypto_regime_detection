"""Long/cash accounting with lagged signals, price drift, entry and exit costs."""
import numpy as np
import pandas as pd


def backtest(prices, signal, cost_bps=10, delay=2):
    prices = prices.astype(float)
    if prices.isna().any() or prices.le(0).any():
        raise ValueError("Backtest requires valid open prices on every valuation date")
    if signal.dropna().lt(0).any() or signal.dropna().gt(1).any():
        raise ValueError("Long/cash target weights must be between zero and one")
    desired = signal.reindex(prices.index).shift(delay).fillna(0).to_numpy(copy=True)
    desired[-1] = 0  # Explicit liquidation at final available open; no future price.
    cost = cost_bps / 10_000
    asset_value, cash, previous_equity = 0., 1., 1.
    rows = []
    values = prices.to_numpy()
    for i, weight in enumerate(desired):
        if i:
            asset_value *= values[i] / values[i-1]
        equity = asset_value + cash
        gap = weight * equity - asset_value
        trade = gap / (1 + cost * weight * np.sign(gap))
        fee = cost * abs(trade)
        asset_value += trade
        cash -= trade + fee
        equity_after = asset_value + cash
        rows.append({"equity": equity_after, "net_return": equity_after / previous_equity - 1,
                     "weight": asset_value / equity_after, "turnover": abs(trade) / equity,
                     "fee": fee / previous_equity})
        previous_equity = equity_after
    result = pd.DataFrame(rows, index=prices.index)
    result["drawdown"] = result.equity / result.equity.cummax().clip(lower=1) - 1
    return result


def performance(result):
    returns = result.net_return
    days = len(result) - 1
    volatility = returns.std(ddof=1) * np.sqrt(365)
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2)) * np.sqrt(365)
    cagr = result.equity.iloc[-1] ** (365 / max(days, 1)) - 1
    return {"days": len(result), "cagr": cagr, "volatility": volatility,
            "sharpe": returns.mean() * 365 / volatility if volatility else float("nan"),
            "sortino": returns.mean() * 365 / downside if downside else float("nan"),
            "max_drawdown": result.drawdown.min(), "turnover": result.turnover.sum(),
            "total_fees_fraction": result.fee.sum(), "average_exposure": result.weight.mean(),
            "ending_value": result.equity.iloc[-1], "risk_free_assumption": 0.}


def paired_bootstrap(left, right, repetitions=1000, block=30, seed=2026):
    a, b = np.asarray(left), np.asarray(right)
    if len(a) != len(b):
        raise ValueError("Paired bootstrap must use the same dates")
    rng, n = np.random.default_rng(seed), len(a)
    differences = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, int(np.ceil(n / block)))
        index = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        def sharpe(values):
            sd = values.std(ddof=1)
            return np.sqrt(365) * values.mean() / sd if sd else np.nan
        differences.append(sharpe(a[index]) - sharpe(b[index]))
    return {"sharpe_difference_ci_low": float(np.nanquantile(differences, .025)),
            "sharpe_difference_ci_high": float(np.nanquantile(differences, .975)),
            "bootstrap_repetitions": repetitions, "block_days": block}


def portfolio_comparison(frame, predictions, start, end, costs=(5, 10, 25), target=.4):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    dates = pd.date_range(start, end, name="date")
    prices = frame.open.reindex(dates)
    volatility = (target / frame.rv30).clip(0, 1).reindex(dates)
    regime = volatility * (1 - predictions.p_high_vol.reindex(dates)).fillna(0)
    targets = {"buy_hold": pd.Series(1., index=dates), "vol_target": volatility,
               "regime_aware": regime}
    tables, paths = [], {}
    for cost in costs:
        for name, signal in targets.items():
            result = backtest(prices, signal, cost_bps=cost, delay=2)
            tables.append({"strategy": name, "cost_bps": cost, **performance(result)})
            if cost == 10:
                paths[name] = result
    uncertainty = paired_bootstrap(paths["regime_aware"].net_return, paths["vol_target"].net_return)
    return pd.DataFrame(tables), paths, uncertainty
