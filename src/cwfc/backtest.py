"""Rolling-origin evaluation inside the training window, model selection, residual intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.forecasters import BASELINE, History
from cwfc.metrics import mae, rmse, smape

PEAK_QUANTILE = 0.75  # "high load" = above the 75th percentile of the history known at the origin
INTERVAL = (0.05, 0.95)
HORIZON_BUCKET = 6  # residual quantiles are pooled over 30-minute blocks of lead time


def rolling_origins(n_train: int, horizon: int, min_train: int = 48, stride: int = 2) -> list[int]:
    """Forecast origins whose full horizon fits inside the training window."""
    last = n_train - horizon
    first = min(min_train, last)
    if first < 2:
        return []
    return list(range(first, last + 1, stride))


def run_backtest(hist: History, models: dict, horizon: int, origins: list[int]) -> pd.DataFrame:
    rows = []
    for origin in origins:
        past = hist.upto(origin)
        actual = hist.y[origin:origin + horizon]
        peak_threshold = np.quantile(past.y, PEAK_QUANTILE)
        for name, model in models.items():
            forecast = model(past, horizon)
            rows.append(pd.DataFrame({
                "model": name,
                "origin": origin,
                "h": np.arange(1, horizon + 1),
                "timestamp": hist.timestamps[origin:origin + horizon],
                "actual": actual,
                "forecast": forecast,
                "peak": actual > peak_threshold,
            }))
    return pd.concat(rows, ignore_index=True)


def summarise(bt: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    out = []
    for name, g in bt.groupby("model", sort=False):
        err = g["forecast"] - g["actual"]
        peak = g[g["peak"]]
        out.append({
            "model": name,
            "mae": mae(g["actual"], g["forecast"]),
            "rmse": rmse(g["actual"], g["forecast"]),
            "smape_pct": smape(g["actual"], g["forecast"]),
            "bias": float(err.mean()),
            "peak_mae": mae(peak["actual"], peak["forecast"]) if len(peak) else np.nan,
            "peak_bias": float((peak["forecast"] - peak["actual"]).mean()) if len(peak) else np.nan,
            "n_peak": len(peak),
            "mae_h1_6": float(err[g["h"] <= 6].abs().mean()),
            "mae_h19_24": float(err[g["h"] >= 19].abs().mean()),
        })
    table = pd.DataFrame(out)
    base = table.set_index("model").loc[baseline]
    for col in ["mae", "rmse", "peak_mae"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            table[f"{col}_vs_baseline"] = np.where(base[col] == 0, np.where(table[col] == 0, 1.0, np.inf), table[col] / base[col])
    return table


def select_model(summary: pd.DataFrame) -> str:
    """Lowest validation MAE; ties go to the earlier (simpler) candidate."""
    return str(summary.loc[summary["mae"].idxmin(), "model"])


def residual_quantiles(bt: pd.DataFrame, model: str, horizon: int) -> pd.DataFrame:
    """Per lead-time bucket quantiles of (actual - forecast) for the chosen model."""
    g = bt[bt["model"] == model]
    resid = g["actual"] - g["forecast"]
    bucket = (g["h"] - 1) // HORIZON_BUCKET
    q = resid.groupby(bucket).quantile(list(INTERVAL)).unstack()
    q.columns = ["q_low", "q_high"]
    h = np.arange(1, horizon + 1)
    return q.reindex((h - 1) // HORIZON_BUCKET).set_index(pd.Index(h, name="h")).fillna(0.0)
