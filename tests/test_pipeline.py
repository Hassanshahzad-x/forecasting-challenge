import numpy as np
import pandas as pd

from cwfc.backtest import rolling_origins, run_backtest
from cwfc.data import build_panel, split_index
from cwfc.evaluate import validate_predictions
from cwfc.forecasters import candidates, make_history, seasonal_ar1
from cwfc.predict import main as predict_main


def test_targets_handle_duplicates_invalid_rows_and_gaps():
    raw = pd.DataFrame({
        "timestamp": [0, 0, 0, 300, 300, 900, 900],
        "vm_id": ["a", "a", "b", "a", "b", "a", "b"],
        "avg_cpu": [10.0, 20.0, 30.0, np.nan, 50.0, 101.0, 4.0],
    })
    panel = build_panel(raw)
    t = panel.targets()
    assert t["timestamp"].tolist() == [0, 300, 600, 900]
    assert t["active_vm_count"].tolist() == [2, 1, 0, 1]  # duplicate counted once, NaN/101 dropped
    assert t["aggregate_cpu_utilization"].iloc[0] == 22.5  # mean(mean(10, 20), 30)
    assert np.isnan(t["aggregate_cpu_utilization"].iloc[2])  # empty slot is not invented
    assert panel.quality["duplicate_rows"] == 1
    assert panel.quality["invalid_rows"] == 2


def test_backtest_forecasts_ignore_data_after_origin():
    rng = np.random.default_rng(1)
    ts = np.arange(110) * 300
    y = 10 + rng.normal(size=110)
    cpu = rng.uniform(0, 20, size=(50, 110))
    models = candidates("aggregate_cpu_utilization", has_vm_matrix=True)
    origins = rolling_origins(110, 24)
    a = run_backtest(make_history(ts, y, 300, cpu), models, 24, origins)
    # Perturb everything after the first origin's history; its forecasts must not change.
    y2, cpu2 = y.copy(), cpu.copy()
    y2[origins[0]:] += 50
    cpu2[:, origins[0]:] += 50
    b = run_backtest(make_history(ts, y2, 300, cpu2), models, 24, origins)
    first = a["origin"] == origins[0]
    assert np.allclose(a.loc[first, "forecast"], b.loc[first, "forecast"])


def test_seasonal_ar1_recovers_hourly_pattern():
    ts = np.arange(110) * 300
    pattern = np.sin(2 * np.pi * np.arange(12) / 12)
    y = 9 + pattern[np.arange(110) % 12]
    out = seasonal_ar1(make_history(ts, y, 300), 24)
    expected = 9 + pattern[np.arange(110, 134) % 12]
    assert np.allclose(out, expected, atol=1e-6)


def test_end_to_end_on_small_irregular_dataset(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for vm in range(40):
        for step in range(60):
            if rng.random() < 0.05:  # random missing readings, so VM counts vary
                continue
            rows.append((1200 + step * 300, f"vm{vm}", rng.uniform(0, 60)))
    raw = pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])
    raw["min_cpu"], raw["max_cpu"] = raw["avg_cpu"] * 0.5, raw["avg_cpu"] * 1.5
    raw.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    out = tmp_path / "predictions.csv"

    predict_main(["--data", str(tmp_path), "--output", str(out)])

    pred = pd.read_csv(out)
    holdout = np.sort(raw["timestamp"].unique())[split_index(60):]
    validate_predictions(pred, holdout)
