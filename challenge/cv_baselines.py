"""Validation harness and linear baseline ladder.

Repeated 5-fold CV (R^2) over pipelines whose preprocessing lives inside
the fold, so imputation/scaling never leak across folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA = Path(__file__).parent / "data" / "summer2026_kaggle_linear_regression_challenge_train.csv"

train = pd.read_csv(DATA)
X = train.drop(columns=["target", "Id"])
y = train["target"]

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)

alphas = np.logspace(-3, 3, 25)
models = {
    "median-impute + LinearRegression": make_pipeline(
        SimpleImputer(strategy="median"), LinearRegression()
    ),
    "median-impute + scale + RidgeCV": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=alphas)
    ),
    "median-impute + scale + LassoCV": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LassoCV(alphas=alphas, random_state=SEED, max_iter=50_000),
    ),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    print(f"{name:40s} R2 = {scores.mean():.4f} +/- {scores.std():.4f}")

lasso = models["median-impute + scale + LassoCV"].fit(X, y)
coef = pd.Series(lasso[-1].coef_, index=X.columns)
print(f"\nLassoCV alpha = {lasso[-1].alpha_:.4f}")
print("zeroed features:", list(coef[coef == 0].index) or "none")
print("standardized |coef| ranking:")
print(coef.abs().sort_values(ascending=False).round(2).to_string())
