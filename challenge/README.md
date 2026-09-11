# Kaggle: Sofia ML Regression (Summer 2026) — Solution Notes

[Lin_Haoyan](https://www.kaggle.com/haoyanlin3) · metric: R² · final score: 0.57186 public / 0.54051 private

The [competition](https://www.kaggle.com/competitions/sofia-ml-regression-2026-summer-private/overview)
is a tabular regression task: predict a continuous `target` from 20 anonymous numeric
features (`x0..x19`), with 2500 rows to train on and 2500 to predict, scored by R².

## TL;DR

Those 20 features turned out to relate to the target in a straight line, buried in noise
with rare but huge outliers. So the final model is deliberately simple: fill missing
values with the column median, fit plain linear regression, and drop five features that
carry no signal (x1, x5, x8, x9, x11).

The real work was testing ideas fairly: every candidate was scored on the same
cross-validation folds, so even tiny differences could be trusted. Judged that way, every
fancier approach lost to plain linear regression, and dropping the five dead features was
the only change that helped (CV R² 0.5348 to 0.5369, winning all 15 folds). The private
score, 0.54051, landed almost exactly on that estimate — the validation was honest, and
the model sits near the noise ceiling of this dataset.

## EDA

| Fact | Value |
|---|---|
| Size | 2500 train and 2500 test rows; 20 features (`x0..x19`) + `target` + `Id` |
| Missing values | about 1% of feature cells; the target column is complete |
| Features | roughly centered, very different scales |
| Correlations | features are essentially uncorrelated with each other (max 0.056) |
| Target | mean 1.8, std 149, range −1215 to 968 |

Fitting a straight-line model and studying what's left over tells most of the story: the
leftover noise is large (std ≈ 101 against a target std of 149), perfectly symmetric, but
far more outlier-prone than normal noise — the worst point sits eleven standard
deviations out. Linear signal, heavy-tailed noise; the experiments below all make sense
in that light.

## Experiments

In the order they happened:

| What was tried | What happened |
|---|---|
| A fair test bench: 5-fold cross-validation repeated 3 times, same folds for every model, preprocessing refit inside each fold | tells apart models only ~0.002 R² apart; used for everything below |
| Baselines: median fill + linear regression, plus ridge and lasso (`cv_baselines.py`) | 0.5348, all three essentially tied — regularization had nothing to fix |
| Curved fits: squared and interaction terms, gradient boosting | 0.45–0.52, all clearly worse — the relationship really is a straight line |
| Outlier-resistant fitting (Huber loss) | ~0.48, worse — the outliers are symmetric noise, so ignoring them just throws away data |
| Smarter missing-value handling, weighted fitting, training only on complete rows | all matched or lost to the simple median fill |
| Dropping weak features step by step, in the order lasso ranked them (`cv_prune.py`) | dropping x1, x5, x8, x9, x11 gained +0.0021 and won all 15 folds — the final model |

## Submissions

| Tag | Model | CV R² | Public R² |
|---|---|---|---|
| `01_baseline_linear` | median fill + linear regression, all 20 features | 0.5348 | 0.57121 |
| `02_pruned_linear` | median fill + linear regression, 5 features dropped | 0.5369 | 0.57186 |

## Code

Everything lives at <https://github.com/HYLTE/ml-hw-summer-2026-python/tree/main/challenge>.

The final score comes from [`make_submission.py`](make_submission.py): running
`python make_submission.py 02_pruned_linear` rebuilds the uploaded file byte for byte.
[`cv_baselines.py`](cv_baselines.py) is the test bench and baseline comparison;
[`cv_prune.py`](cv_prune.py) is the feature-dropping sweep that picked the final model.

To reproduce: `uv sync` at the repo root, put the competition CSVs in `challenge/data/`
(not committed, per the data rules), then `uv run python challenge/<script>.py`.
