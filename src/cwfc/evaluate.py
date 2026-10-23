"""Score predictions.csv against the 24-step holdout (run only after predicting).

    python -m cwfc.evaluate --data challenge_data --predictions predictions.csv \
        --artifacts artifacts --output reports

Nothing here feeds back into model selection: the model was fixed by `cwfc.predict` using
the training window only. Holdout scores of the other candidates are printed for
transparency, not for choosing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cwfc.backtest import PEAK_QUANTILE, rolling_origins, run_backtest
from cwfc.data import HOLDOUT_STEPS, METADATA_FILE, build_panel, load_metadata, load_readings, split_index
from cwfc.forecasters import BASELINE, candidates, make_history, persistence, seasonal_ar1
from cwfc.metrics import mae, rmse, smape
from cwfc.predict import TARGET_BOUNDS

COLUMNS = ["timestamp", *TARGET_BOUNDS]
POLICY_QUANTILES = [None, 0.5, 0.8, 0.9, 0.95, 0.99]  # None = no margin


def validate_predictions(pred: pd.DataFrame, holdout_timestamps) -> None:
    """Enforce the SUBMISSION.md contract."""
    assert list(pred.columns) == COLUMNS, f"columns must be {COLUMNS}"
    assert len(pred) == len(holdout_timestamps), "wrong number of rows"
    assert pred["timestamp"].is_unique, "duplicate timestamps"
    assert (pred["timestamp"].to_numpy() == np.asarray(holdout_timestamps)).all(), "timestamps differ from holdout"
    values = pred[list(TARGET_BOUNDS)].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "non-finite predictions"
    assert (pred["active_vm_count"] >= 0).all(), "negative VM count"
    assert pred["aggregate_cpu_utilization"].between(0, 100).all(), "CPU outside [0,100]"


def scores(actual, forecast, peak) -> dict:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    out = {
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "smape_pct": smape(actual, forecast),
        "bias": float(np.mean(forecast - actual)),
        "max_abs_error": float(np.max(np.abs(forecast - actual))),
        "n_peak": int(peak.sum()),
    }
    out["peak_mae"] = mae(actual[peak], forecast[peak]) if peak.any() else np.nan
    return out


def policy_table(residuals: np.ndarray, forecast, actual) -> pd.DataFrame:
    """Provision = forecast + margin, margin = quantile of CV residuals (actual - forecast)."""
    rows = []
    for q in POLICY_QUANTILES:
        margin = 0.0 if q is None else float(np.quantile(residuals, q))
        provision = np.asarray(forecast) + margin
        gap = provision - np.asarray(actual)
        rows.append({
            "margin_quantile": "none" if q is None else q,
            "margin_pp": margin,
            "under_rate": float(np.mean(gap < 0)),
            "mean_shortfall_pp": float(-gap[gap < 0].mean()) if (gap < 0).any() else 0.0,
            "mean_headroom_pp": float(gap[gap >= 0].mean()) if (gap >= 0).any() else 0.0,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data", type=Path)
    parser.add_argument("--predictions", default="predictions.csv", type=Path)
    parser.add_argument("--artifacts", default="artifacts", type=Path)
    parser.add_argument("--output", default="reports", type=Path)
    args = parser.parse_args(argv)
    fig_dir = args.output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(load_readings(args.data))
    targets = panel.targets()
    split = split_index(len(targets), HOLDOUT_STEPS)
    train, hold = targets.iloc[:split], targets.iloc[split:]
    pred = pd.read_csv(args.predictions)
    validate_predictions(pred, hold["timestamp"])
    selection = json.loads((args.artifacts / "selection.json").read_text())

    rows, policy = [], []
    for target in TARGET_BOUNDS:
        hist = make_history(train["timestamp"].to_numpy(), train[target].to_numpy(), panel.step, panel.cpu[:, :split])
        actual = hold[target].to_numpy()
        peak = actual > np.quantile(train[target], PEAK_QUANTILE)
        chosen = selection[target]["model"]
        rows.append({"target": target, "model": f"submitted ({chosen})", **scores(actual, pred[target], peak)})
        for name, model in candidates(target, has_vm_matrix=True).items():
            if name != chosen:
                rows.append({"target": target, "model": name, **scores(actual, model(hist, HOLDOUT_STEPS), peak)})

        bt_path = args.artifacts / f"backtest_{target}.csv"
        if target == "aggregate_cpu_utilization" and bt_path.exists():
            bt = pd.read_csv(bt_path)
            sel = bt[bt["model"] == chosen]
            resid = (sel["actual"] - sel["forecast"]).to_numpy()
            for split_name, f, a in [("cv", sel["forecast"], sel["actual"]), ("holdout", pred[target], actual)]:
                policy.append(policy_table(resid, f, a).assign(evaluated_on=split_name))
            _plot_cv(bt, fig_dir / "cv_mae_by_horizon.png")
            _plot_holdout(train, hold, pred, args.artifacts / "forecast_intervals.csv", fig_dir / "holdout_forecast.png")

    table = pd.DataFrame(rows)
    base = table[table["model"] == BASELINE].set_index("target")["mae"]
    table["mae_vs_persistence"] = table["mae"] / table["target"].map(base).replace(0, np.nan)
    table.to_csv(args.output / "holdout_metrics.csv", index=False)
    print(table.round(4).to_string(index=False))

    intervals_path = args.artifacts / "forecast_intervals.csv"
    if intervals_path.exists():
        iv = pd.read_csv(intervals_path)
        iv = iv[iv["target"] == "aggregate_cpu_utilization"]
        a = hold["aggregate_cpu_utilization"].to_numpy()
        cover = float(np.mean((a >= iv["lower_90"].to_numpy()) & (a <= iv["upper_90"].to_numpy())))
        print(f"\n90% interval empirical coverage on holdout: {cover:.3f}")
    if policy:
        pt = pd.concat(policy, ignore_index=True)
        pt.to_csv(args.output / "policy_tradeoff.csv", index=False)
        print("\n" + pt.round(4).to_string(index=False))

    if (args.data / METADATA_FILE).exists():
        seg = segment_study(panel, load_metadata(args.data), split)
        seg.to_csv(args.output / "segment_study.csv", index=False)
        print("\n" + seg.round(4).to_string(index=False))


def segment_study(panel, meta: pd.DataFrame, split: int, column: str = "vm_category") -> pd.DataFrame:
    """Does the seasonal AR(1) gain hold per segment, and does bottom-up beat the direct model?

    CV uses the training window only; holdout columns are scored once for reporting.
    """
    labels = meta.set_index("vm_id")[column].reindex(panel.vm_ids).astype(str).to_numpy()
    models = {"persistence": persistence, "seasonal_ar1": seasonal_ar1}
    origins = rolling_origins(split, HOLDOUT_STEPS)
    rows, bottom_up = [], {"cv": 0.0, "holdout": 0.0}
    total = panel.targets()["aggregate_cpu_utilization"].to_numpy()
    for lab in sorted(set(labels)):
        members = labels == lab
        with np.errstate(invalid="ignore"):
            y = np.nanmean(panel.cpu[members], axis=0)
        hist = make_history(panel.timestamps[:split], y[:split], panel.step)
        bt = run_backtest(hist, models, HOLDOUT_STEPS, origins)
        hold_fc = {name: m(hist, HOLDOUT_STEPS) for name, m in models.items()}
        for name in models:
            sel = bt[bt["model"] == name]
            rows.append({"segment": lab, "vms": int(members.sum()), "model": name,
                         "cv_mae": mae(sel["actual"], sel["forecast"]),
                         "holdout_mae": mae(y[split:], hold_fc[name])})
        weight = members.sum() / len(labels)
        ar = bt[bt["model"] == "seasonal_ar1"]
        bottom_up["cv"] = bottom_up["cv"] + weight * ar["forecast"].to_numpy()
        bottom_up["holdout"] = bottom_up["holdout"] + weight * hold_fc["seasonal_ar1"]
    cv_actual = np.concatenate([total[o:o + HOLDOUT_STEPS] for o in origins])
    rows.append({"segment": "ALL (bottom-up sum of segments)", "vms": len(labels), "model": "seasonal_ar1",
                 "cv_mae": mae(cv_actual, bottom_up["cv"]), "holdout_mae": mae(total[split:], bottom_up["holdout"])})
    return pd.DataFrame(rows)


def _plot_cv(bt: pd.DataFrame, path: Path) -> None:
    err = bt.assign(abs_err=(bt["forecast"] - bt["actual"]).abs())
    curve = err.groupby(["model", "h"])["abs_err"].mean().unstack(0)
    ax = curve.plot(figsize=(8, 4.5), marker=".")
    ax.set(xlabel="lead time (5-min steps)", ylabel="MAE (pp CPU)", title="Rolling-origin CV: MAE by lead time")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_holdout(train, hold, pred, intervals_path: Path, path: Path) -> None:
    hours = lambda ts: np.asarray(ts) / 3600
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hours(train["timestamp"]), train["aggregate_cpu_utilization"], color="0.4", label="training actual")
    ax.plot(hours(hold["timestamp"]), hold["aggregate_cpu_utilization"], color="k", label="holdout actual")
    ax.plot(hours(pred["timestamp"]), pred["aggregate_cpu_utilization"], color="C3", label="forecast")
    ax.axhline(train["aggregate_cpu_utilization"].iloc[-1], color="C0", ls="--", lw=1, label="persistence")
    if intervals_path.exists():
        iv = pd.read_csv(intervals_path)
        iv = iv[iv["target"] == "aggregate_cpu_utilization"]
        ax.fill_between(hours(iv["timestamp"]), iv["lower_90"], iv["upper_90"], color="C3", alpha=0.2, label="90% interval (CV residuals)")
    ax.set(xlabel="hours since trace start", ylabel="mean VM avg_cpu (%)", title="Holdout forecast: aggregate CPU utilization")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
