"""Loading, cleaning and target construction for the VM CPU readings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

READINGS_FILE = "vm_cpu_readings.csv.gz"
METADATA_FILE = "vm_metadata.csv.gz"
REQUIRED_COLUMNS = ["timestamp", "vm_id", "avg_cpu"]
HOLDOUT_STEPS = 24


@dataclass
class Panel:
    """VM x time matrix of avg_cpu on a regular grid (NaN = no valid reading)."""

    timestamps: np.ndarray  # int seconds, regular grid
    vm_ids: np.ndarray
    cpu: np.ndarray  # shape (n_vm, n_time)
    step: int
    quality: dict = field(default_factory=dict)

    def targets(self) -> pd.DataFrame:
        observed = ~np.isnan(self.cpu)
        count = observed.sum(axis=0)
        total = np.where(observed, self.cpu, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_cpu = np.where(count > 0, total / count, np.nan)
        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "active_vm_count": count.astype(int),
                "aggregate_cpu_utilization": mean_cpu,
            }
        )


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / READINGS_FILE)


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / METADATA_FILE)


def build_panel(raw: pd.DataFrame) -> Panel:
    """Clean raw readings and pivot them onto a regular five-minute grid.

    Rules (all counted in `quality`):
    - rows with a missing/non-finite `avg_cpu` or one outside [0, 100] are invalid and dropped;
    - timestamps are snapped to the nearest grid point (grid step = most common spacing);
    - duplicate (vm_id, timestamp) readings are collapsed to their mean, so a VM counts once;
    - grid timestamps without any valid reading are kept (count 0, CPU NaN) rather than dropped.
    """
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"readings are missing columns: {sorted(missing)}")
    df = raw[REQUIRED_COLUMNS].copy()
    df["avg_cpu"] = pd.to_numeric(df["avg_cpu"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    valid = (
        np.isfinite(df["avg_cpu"])
        & df["avg_cpu"].between(0, 100)
        & np.isfinite(df["timestamp"])
        & df["vm_id"].notna()
    )
    quality = {"rows": len(df), "invalid_rows": int((~valid).sum())}
    df = df[valid]
    if df.empty:
        raise ValueError("no valid CPU readings")

    unique_ts = np.unique(df["timestamp"].to_numpy())
    diffs = np.diff(unique_ts)
    step = int(pd.Series(diffs).mode().iloc[0]) if len(diffs) else 300
    t0 = int(unique_ts[0])
    slot = np.rint((df["timestamp"].to_numpy() - t0) / step).astype(int)
    quality["off_grid_rows"] = int((t0 + slot * step != df["timestamp"].to_numpy()).sum())
    df = df.assign(slot=slot)
    quality["duplicate_rows"] = int(df.duplicated(["vm_id", "slot"]).sum())
    df = df.groupby(["vm_id", "slot"], as_index=False, sort=False)["avg_cpu"].mean()

    n_time = int(slot.max()) + 1
    vm_codes, vm_ids = pd.factorize(df["vm_id"])
    cpu = np.full((len(vm_ids), n_time), np.nan)
    cpu[vm_codes, df["slot"].to_numpy()] = df["avg_cpu"].to_numpy()
    quality["empty_timestamps"] = int((np.isnan(cpu).all(axis=0)).sum())
    quality["step_seconds"] = step
    quality["n_vms"] = len(vm_ids)
    quality["n_timestamps"] = n_time
    return Panel(t0 + step * np.arange(n_time), np.asarray(vm_ids), cpu, step, quality)


def split_index(n_timestamps: int, holdout: int = HOLDOUT_STEPS) -> int:
    """Index of the first holdout timestamp (110 for the 134-step challenge grid)."""
    if n_timestamps <= holdout:
        raise ValueError(f"need more than {holdout} timestamps, got {n_timestamps}")
    return n_timestamps - holdout
