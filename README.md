# Early forecasting of irrigation demand from IoT greenhouse time series

Code for the paper:

> **Early Forecasting of Irrigation Demand from Multivariate IoT Greenhouse Time Series under Strict Chronological Evaluation**
> B. El Khalyly and Z. Ibn Batouta, 2026 (under review).

The paper asks whether an irrigation event can be predicted 10, 30 or 60 minutes before it happens, using only the five sensor streams already logged by a low-cost greenhouse controller.
Everything here runs from the raw CSV, and every number in the paper's tables comes out of these scripts.

## What the pipeline does

1. **`src/prep2.py`** aggregates the labelled records to a one-minute grid (sensor means, label maximum). It splits the series into acquisition segments wherever two consecutive minutes are more than 2.1 min apart. Within each segment it builds 37 causal predictors:
   - the 5 current readings;
   - 5- and 15-min trailing means and standard deviations;
   - 1- and 5-min differences;
   - sine/cosine time of day.

   It then builds the three targets `y10`, `y30`, `y60`. Minutes that are already irrigating are dropped, because the task is early warning, not detection.
2. **`src/eval2.py`** trains six cost-sensitive classifiers on data before 19 May 2026. It picks one decision threshold per model and horizon on 19 May (validation) and scores 20–24 May once. The same script runs the current-readings-only ablation.
3. **`src/bootstrap2.py`** computes 2,000-replicate percentile intervals for the logistic-regression test predictions.
4. **`src/figs2.py`** draws the three figures.

Nothing from the test period is used for scaling, fitting, or threshold choice.

## Run it in Google Colab or Jupyter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/badrelkhalylyphd-boop/irrigation-early-warning-forecasting/blob/main/reproduce_colab.ipynb)

`reproduce_colab.ipynb` runs the whole experiment in one notebook. It installs the pinned library versions; on Colab the session restarts once, after which you choose *Run all* again. On Colab, first upload `greenhouse_dataset.csv` in the Files panel (it is read from `/content`; if it is missing the notebook asks for it). Locally, the copy in `data/` is used, and ends with a table that compares every number reported in the paper with the current run and marks each one ✅ or ❌.


## Quick start (command line)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

The run takes about two minutes on a laptop. Outputs land in `outputs/`:

| File | Content | Paper |
|---|---|---|
| `forecast_features.csv` | 7,090 eligible minutes × 37 features + targets | Sec. III |
| `forecast_results.json` | all models, all horizons, ablation, thresholds, confusion matrices | Tables III–V, VII; Fig. 1 |
| `bootstrap_results.json` | 95% bootstrap intervals | Table VI |
| `figures/*.png` | Figs. 1–3 | Figs. 1–3 |

To run the steps one at a time from inside `outputs/`:

```bash
python ../src/prep2.py ../data/greenhouse_dataset.csv --out forecast_features.csv
python ../src/eval2.py forecast_features.csv --out forecast_results.json
python ../src/bootstrap2.py forecast_features.csv --out bootstrap_results.json --nboot 2000
python ../src/figs2.py forecast_features.csv forecast_results.json --outdir figures
```

## Expected results

A copy of the outputs we obtained is in `results/` so you can diff your run against it.

**Split sizes** (positives / samples)

| Horizon | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| 10 min | 125 / 3,587 | 84 / 1,312 | 340 / 2,191 | 15.52% |
| 30 min | 246 / 3,587 | 204 / 1,312 | 507 / 2,191 | 23.14% |
| 60 min | 315 / 3,587 | 336 / 1,312 | 560 / 2,191 | 25.56% |

**Logistic regression on the held-out test days (%)**

| Horizon | F1 | MCC | PR-AUC | 95% CI of F1 |
|---|---|---|---|---|
| 10 min | 59.74 | 51.82 | 47.22 | 55.42–63.65 |
| 30 min | 63.42 | 55.40 | 64.06 | 59.74–66.89 |
| 60 min | 73.42 | 65.43 | 60.78 | 70.26–76.40 |

At 60 min the confusion matrix is TN = 1,529, FP = 102, FN = 176, TP = 384.
With the five current readings only, the 60-min F1 drops to 40.90%.

## Reproducibility notes

- Verified on Windows 11, Python 3.11.9, with the exact versions in `requirements.txt`. All seeds are fixed at 42, and the bootstrap seed is 20261016.
- The logistic-regression results are the least sensitive to library versions. The tree ensembles can move by a point or so with other versions of scikit-learn, XGBoost or CatBoost. Install the pinned versions if you want identical tables.
- Random Forest and Extra Trees run multi-threaded, so their PR-AUC can differ around the fifth decimal between runs. Reported values (two decimals) are unaffected.
- The selected thresholds for logistic regression sit very close to 1 (e.g. 0.999991 at 60 min). This is a side effect of balanced class weights on a standardized feature space, not a calibrated probability. See Sec. IV-B of the paper.
- At 10 and 30 min, several boosted models pick a near-zero threshold on the validation day and end up labelling almost every test minute as positive (MCC ≈ 0). This behaviour is reported as is; it is part of why the linear model wins.

## Data

See [`data/README.md`](data/README.md). The dataset is © its author and distributed under CC BY 4.0.

## Citation

If you use this code, please cite the paper (reference to be updated on publication) and the dataset:

```
I. Keok, "IoT Greenhouse Irrigation Sensor Dataset," Mendeley Data, V1, 2026, doi: 10.17632/2tvpsht4s4.1.
```

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the original authors.
