# Cloud Workload Forecasting Challenge: Solution

Forecasts for the 2-hour holdout of the Azure Public Dataset V2 challenge release
(`data-v1`, 10,000 VMs x 134 five-minute steps).

- Report: [`reports/final_report.md`](reports/final_report.md)
- Workload analysis numbers: [`reports/eda_summary.md`](reports/eda_summary.md) (generated)
- Submission: [`predictions.csv`](predictions.csv)

The original assignment is in [`ASSIGNMENT.md`](ASSIGNMENT.md), [`DATASET.md`](DATASET.md) and
[`SUBMISSION.md`](SUBMISSION.md).

## Results in brief

| Target | Selected model (train-only CV) | Holdout MAE | Persistence MAE |
|---|---|---:|---:|
| `active_vm_count` | persistence | 0 | 0 |
| `aggregate_cpu_utilization` | seasonal AR(1): hourly profile + mean-reverting level | 0.091 pp | 0.147 pp |

`active_vm_count` is 10,000 at every timestamp. The release only contains VMs with complete
134-step traces, so there is no churn in the window. `aggregate_cpu_utilization` is about 9.4%
with a stable one-hour cycle and occasional fleet-wide transients from batch jobs.
The report has the full numbers.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

Dependencies: `numpy`, `pandas`, `scikit-learn` (pooled gradient-boosting candidate) and
`matplotlib` (figures).

## Data

Download `data-v1.tar.gz` from the
[data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract it into `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
(cd challenge_data && shasum -a 256 -c SHA256SUMS)   # or sha256sum -c on Linux
```

## Generate predictions

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv --artifacts artifacts
```

This command:

1. cleans the readings and builds both targets on the five-minute grid;
2. splits off the final 24 timestamps (the holdout). Their targets are not computed;
3. runs a rolling-origin backtest inside the first 110 steps (20 origins, 24-step horizon)
   for every candidate model;
4. selects the model with the lowest backtest MAE, refits it on all 110 training steps and
   forecasts the 24 holdout timestamps;
5. writes `predictions.csv`. With `--artifacts`, it also writes the backtest tables, the
   selected models and the 90% forecast intervals.

The run takes about 5 minutes on a laptop. Most of that time is the pooled GBM candidate.
The pipeline does not depend on the exact shape of the release. It accepts any VM count,
missing readings, duplicates and gaps, and always forecasts the last 24 grid timestamps.
`tests/test_pipeline.py` runs it on a small irregular synthetic dataset.

## Analysis and holdout scoring

```bash
python -m cwfc.eda --data challenge_data --output reports
python -m cwfc.evaluate --data challenge_data --predictions predictions.csv --artifacts artifacts --output reports
```

`cwfc.eda` writes `reports/eda_summary.md` and `reports/figures/{overview,segments}.png`.
Run `cwfc.evaluate` only after predicting. It validates `predictions.csv` against the
submission contract, scores it on the holdout and writes `reports/holdout_metrics.csv`,
`reports/policy_tradeoff.csv` and two figures. Its output never feeds back into model
selection.

## Code layout

| File | Purpose |
|---|---|
| `src/cwfc/data.py` | cleaning, grid construction, target definitions, 110/24 split |
| `src/cwfc/forecasters.py` | baselines, seasonal persistence, seasonal AR(1), pooled VM-level GBM |
| `src/cwfc/backtest.py` | rolling origins, metrics, model selection, residual-based intervals |
| `src/cwfc/predict.py` | CLI that writes `predictions.csv` |
| `src/cwfc/evaluate.py` | CLI for holdout scoring, contract checks and the capacity-policy trade-off |
| `src/cwfc/eda.py` | CLI that writes the workload analysis figures and tables |
| `src/cwfc/{metrics,baselines,validation}.py` | starter utilities. Metrics are reused |

## Main modelling decisions

- **Targets.** `active_vm_count` is the number of distinct VMs with a valid reading
  (`avg_cpu` finite and in [0, 100]) at each timestamp. `aggregate_cpu_utilization` is the
  mean of those readings. Duplicate readings collapse to their mean, so a VM is counted
  once. A grid timestamp with no readings keeps count 0 and CPU NaN, so no value is
  invented. Inside the training history only, such gaps are linearly interpolated before
  model fitting. The release has no gaps, duplicates or invalid rows.
- **Leakage controls.** Forecasters receive a `History` that is truncated at the origin.
  The holdout targets are never computed in `cwfc.predict`. Metadata CPU statistics
  (`avg_cpu`, `max_cpu`, `p95_max_cpu`) and deletion times describe the full 30-day trace.
  They therefore contain future information and are not used as features.
  `tests/test_pipeline.py` checks that changing data after an origin does not change that
  origin's forecasts.
- **Calendar feature.** The phase within the hour, `(timestamp // 300) % 12`, is known for
  any future timestamp, so it is valid at the forecast origin. The trace has no absolute
  clock time, and 11 hours of data cannot support daily or weekly seasonality.
- **Selection.** The model with the lowest MAE over the 20-origin backtest is selected. The
  candidate list runs from simplest to most complex, and a tie goes to the simpler model.
