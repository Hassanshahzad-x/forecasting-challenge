"""Generate predictions.csv for the 24-step holdout.

    python -m cwfc.predict --data challenge_data --output predictions.csv --artifacts artifacts

Only the training window (all timestamps before the final 24) is passed to validation,
model selection and the final fit. The holdout targets are never computed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.backtest import residual_quantiles, rolling_origins, run_backtest, select_model, summarise
from cwfc.data import HOLDOUT_STEPS, Panel, build_panel, load_readings, split_index
from cwfc.forecasters import BASELINE, candidates, make_history

TARGET_BOUNDS = {
    "active_vm_count": (0.0, np.inf),
    "aggregate_cpu_utilization": (0.0, 100.0),
}


def forecast(panel: Panel, horizon: int = HOLDOUT_STEPS, artifacts: Path | None = None) -> pd.DataFrame:
    split = split_index(len(panel.timestamps), horizon)
    train_cpu = panel.cpu[:, :split]
    train = Panel(panel.timestamps[:split], panel.vm_ids, train_cpu, panel.step).targets()
    out = pd.DataFrame({"timestamp": panel.timestamps[split:]})
    intervals, selection = [], {"train_steps": split, "horizon": horizon, "quality": panel.quality}

    for target, (lo, hi) in TARGET_BOUNDS.items():
        hist = make_history(train["timestamp"].to_numpy(), train[target].to_numpy(), panel.step, train_cpu)
        models = candidates(target, has_vm_matrix=True)
        origins = rolling_origins(split, horizon)
        chosen, quant = BASELINE, None
        if origins:
            bt = run_backtest(hist, models, horizon, origins)
            summary = summarise(bt)
            chosen = select_model(summary)
            quant = residual_quantiles(bt, chosen, horizon)
            if artifacts:
                bt.to_csv(artifacts / f"backtest_{target}.csv", index=False)
                summary.to_csv(artifacts / f"cv_summary_{target}.csv", index=False)
            print(f"\n[{target}] rolling-origin CV, {len(origins)} origins:")
            print(summary[["model", "mae", "rmse", "bias", "peak_mae", "mae_vs_baseline"]].round(4).to_string(index=False))
        point = np.clip(models[chosen](hist, horizon), lo, hi)
        out[target] = point
        selection[target] = {"model": chosen, "cv_origins": origins}
        print(f"[{target}] selected: {chosen}")
        if quant is not None:
            intervals.append(pd.DataFrame({
                "timestamp": out["timestamp"], "target": target, "model": chosen, "forecast": point,
                "lower_90": np.clip(point + quant["q_low"].to_numpy(), lo, hi),
                "upper_90": np.clip(point + quant["q_high"].to_numpy(), lo, hi),
            }))

    if artifacts:
        if intervals:
            pd.concat(intervals).to_csv(artifacts / "forecast_intervals.csv", index=False)
        (artifacts / "selection.json").write_text(json.dumps(selection, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data", type=Path)
    parser.add_argument("--output", default="predictions.csv", type=Path)
    parser.add_argument("--artifacts", type=Path, help="optional dir for CV tables and intervals")
    args = parser.parse_args(argv)
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
    panel = build_panel(load_readings(args.data))
    print("data quality:", panel.quality)
    predictions = forecast(panel, artifacts=args.artifacts)
    predictions.to_csv(args.output, index=False)
    print(f"\nwrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
