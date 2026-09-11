"""How far can feature pruning go? Paired-fold sweep of nested drop-sets
built from the smallest standardized Lasso coefficients (cv_baselines.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

SEED = 42
DATA = Path(__file__).parent / "data" / "summer2026_kaggle_linear_regression_challenge_train.csv"

train = pd.read_csv(DATA)
X = train.drop(columns=["target", "Id"])
y = train["target"].to_numpy()

DROP_SETS = {
    "none (ref)": [],
    "x5,x9,x11": ["x5", "x9", "x11"],
    "+x1": ["x5", "x9", "x11", "x1"],
    "+x1,x8": ["x5", "x9", "x11", "x1", "x8"],
    "+x1,x8,x12": ["x5", "x9", "x11", "x1", "x8", "x12"],
    "+x1,x8,x12,x3": ["x5", "x9", "x11", "x1", "x8", "x12", "x3"],
}

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scores = {name: [] for name in DROP_SETS}

for tr_idx, va_idx in cv.split(X):
    for name, drop in DROP_SETS.items():
        Xs = X.drop(columns=drop).to_numpy()
        imp = SimpleImputer(strategy="median").fit(Xs[tr_idx])
        m = LinearRegression().fit(imp.transform(Xs[tr_idx]), y[tr_idx])
        pred = m.predict(imp.transform(Xs[va_idx]))
        scores[name].append(r2_score(y[va_idx], pred))

base = np.array(scores["none (ref)"])
print(f"{'none (ref)':16s} R2 = {base.mean():.4f}")
for name, vals in scores.items():
    if name == "none (ref)":
        continue
    d = np.array(vals) - base
    print(f"{name:16s} R2 = {np.array(vals).mean():.4f}  delta = {d.mean():+.4f} "
          f"+/- {d.std():.4f}  wins {int((d > 0).sum())}/{len(d)}")
