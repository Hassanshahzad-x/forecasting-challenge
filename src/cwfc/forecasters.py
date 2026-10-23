"""Forecasters for the aggregate targets.

Every forecaster has the signature ``f(hist: History, horizon: int) -> np.ndarray`` and
only ever sees a `History`, which is truncated at the forecast origin. Anything a
forecaster computes (seasonal profile, AR coefficient, GBM fit) is therefore estimated
from information available at the origin.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cwfc.baselines import persistence_forecast, seasonal_naive_forecast

PERIOD_SECONDS = 3600  # the hourly cycle found in the EDA (lag-12 ACF of differences = 0.61)


@dataclass(frozen=True)
class History:
    timestamps: np.ndarray  # int seconds, regular grid, ends at the forecast origin
    y: np.ndarray  # target, gaps linearly interpolated (see `make_history`)
    step: int
    cpu: np.ndarray | None = None  # VM x time avg_cpu matrix, NaN = missing

    def upto(self, n: int) -> "History":
        cpu = None if self.cpu is None else self.cpu[:, :n]
        return History(self.timestamps[:n], self.y[:n], self.step, cpu)

    @property
    def period(self) -> int:
        return max(1, PERIOD_SECONDS // self.step)

    def phase(self, timestamps: np.ndarray) -> np.ndarray:
        return (np.asarray(timestamps) // self.step) % self.period

    def future_timestamps(self, horizon: int) -> np.ndarray:
        return self.timestamps[-1] + self.step * np.arange(1, horizon + 1)


def make_history(timestamps, y, step, cpu=None) -> History:
    """Fill target gaps (timestamps without readings) by linear interpolation.

    Interpolation only ever happens inside the history passed to a forecaster, i.e. it never
    looks past the training window. Leading/trailing gaps take the nearest observed value.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if not ok.any():
        raise ValueError("target has no observed values")
    idx = np.arange(len(y))
    return History(np.asarray(timestamps), np.interp(idx, idx[ok], y[ok]), step, cpu)


# --- baselines ------------------------------------------------------------------------

def _frame(hist: History) -> pd.DataFrame:
    return pd.DataFrame({"y": hist.y})


def persistence(hist: History, horizon: int) -> np.ndarray:
    return persistence_forecast(_frame(hist), pd.DataFrame(index=range(horizon)), "y").to_numpy()


def moving_average_1h(hist: History, horizon: int) -> np.ndarray:
    return np.full(horizon, hist.y[-hist.period:].mean())


def seasonal_naive_1h(hist: History, horizon: int) -> np.ndarray:
    if len(hist.y) < hist.period:
        return persistence(hist, horizon)
    template = pd.DataFrame(index=range(horizon))
    return seasonal_naive_forecast(_frame(hist), template, "y", hist.period).to_numpy()


# --- seasonal statistical models -------------------------------------------------------

def seasonal_profile(hist: History) -> np.ndarray:
    """Mean deviation from a centred one-hour moving average, per phase of the hour."""
    p = hist.period
    if p < 2 or len(hist.y) < 2 * p:
        return np.zeros(p)
    s = pd.Series(hist.y)
    if p % 2 == 0:  # classical 2xP centred moving average
        trend = s.rolling(p).mean().rolling(2).mean().shift(-(p // 2))
    else:
        trend = s.rolling(p, center=True).mean()
    resid = (s - trend).to_numpy()
    prof = pd.Series(resid).groupby(hist.phase(hist.timestamps)).mean()
    prof = prof.reindex(range(p)).fillna(0.0).to_numpy()
    return prof - prof.mean()


def seasonal_persistence(hist: History, horizon: int) -> np.ndarray:
    """Last seasonally adjusted value plus the hourly profile (ablation: no mean reversion)."""
    prof = seasonal_profile(hist)
    level = hist.y[-1] - prof[hist.phase(hist.timestamps[-1])]
    return level + prof[hist.phase(hist.future_timestamps(horizon))]


def seasonal_ar1(hist: History, horizon: int) -> np.ndarray:
    """Hourly profile + AR(1) on the seasonally adjusted series, reverting to its mean.

    level(h) = mu + phi**h * (z_last - mu), with mu the history mean of the adjusted series
    and phi its lag-1 autocorrelation clipped to [0, 1].
    """
    prof = seasonal_profile(hist)
    z = hist.y - prof[hist.phase(hist.timestamps)]
    mu = z.mean()
    dz = z - mu
    denom = float(np.sum(dz[:-1] ** 2))
    phi = 1.0 if denom == 0 else float(np.clip(np.sum(dz[1:] * dz[:-1]) / denom, 0.0, 1.0))
    level = mu + phi ** np.arange(1, horizon + 1) * (z[-1] - mu)
    return level + prof[hist.phase(hist.future_timestamps(horizon))]


# --- pooled VM-level gradient boosting (bottom-up) --------------------------------------

def _vm_features(cpu: np.ndarray, origin: int, target_idx: int, period: int, phase: int, h: int):
    """Per-VM features from columns < origin, for predicting column `target_idx`."""
    past = cpu[:, :origin]
    same_phase = [target_idx - k * period for k in range(1, 4)]
    same_phase = [i for i in same_phase if 0 <= i < origin] or [origin - 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows -> NaN, handled by GBM
        cols = [
            past[:, -1],
            np.nanmean(past[:, -3:], axis=1),
            np.nanmean(past[:, -period:], axis=1),
            np.nanmean(past[:, -3 * period:], axis=1),
            np.nanmean(past, axis=1),
            np.nanstd(past[:, -period:], axis=1),
            past[:, same_phase[0]],
            np.nanmean(past[:, same_phase], axis=1),
        ]
    n = cpu.shape[0]
    return np.column_stack(cols + [np.full(n, h), np.full(n, phase)])


def pooled_gbm(hist: History, horizon: int, n_train_vms: int = 3000, origin_stride: int = 2) -> np.ndarray:
    """Direct multi-horizon GBM trained across VMs; aggregate = mean of VM forecasts.

    Training pairs (origin o, horizon h) all satisfy o + h <= len(history), so labels come
    from the history only. Squared loss targets the conditional mean, which is what averages
    correctly into the aggregate (a median/absolute loss would be biased on skewed VMs).
    """
    cpu, p = hist.cpu, hist.period
    if cpu is None:
        raise ValueError("pooled_gbm needs the VM-level matrix")
    n_time = cpu.shape[1]
    min_hist = min(2 * p, n_time - 1)
    rng = np.random.default_rng(0)
    rows = rng.choice(cpu.shape[0], size=min(n_train_vms, cpu.shape[0]), replace=False)
    sub = cpu[rows]
    phases = hist.phase(hist.timestamps)
    X, y = [], []
    for origin in range(min_hist, n_time, origin_stride):
        for h in range(1, min(horizon, n_time - origin) + 1):
            t = origin + h - 1
            X.append(_vm_features(sub, origin, t, p, phases[t], h))
            y.append(sub[:, t])
    if not X:
        return persistence(hist, horizon)
    X, y = np.vstack(X), np.concatenate(y)
    keep = np.isfinite(y)
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=0)
    model.fit(X[keep], y[keep])

    active = ~np.isnan(cpu[:, -p:]).all(axis=1)  # VMs seen in the last hour are assumed to persist
    future_phase = hist.phase(hist.future_timestamps(horizon))
    out = []
    for h in range(1, horizon + 1):
        feats = _vm_features(cpu[active], n_time, n_time + h - 1, p, future_phase[h - 1], h)
        out.append(model.predict(feats).mean())
    return np.array(out)


BASELINE = "persistence"


def candidates(target: str, has_vm_matrix: bool) -> dict:
    """Candidate set, simplest first: ties in validation go to the earlier (simpler) model."""
    models = {
        "persistence": persistence,
        "moving_average_1h": moving_average_1h,
        "seasonal_naive_1h": seasonal_naive_1h,
        "seasonal_persistence": seasonal_persistence,
        "seasonal_ar1": seasonal_ar1,
    }
    if target == "aggregate_cpu_utilization" and has_vm_matrix:
        models["pooled_gbm"] = pooled_gbm
    return models
