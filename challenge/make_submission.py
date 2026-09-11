"""Generate a submission CSV from a stored model config.

Usage: python make_submission.py [tag]   (default: the newest config)

Submission history (upload each with submission name "Lin_Haoyan"):
  01_baseline_linear   median-impute + OLS, all 20 features        CV 0.5348  public 0.57121
  02_pruned_linear     median-impute + OLS, drop 5 dead features   CV 0.5369  public 0.57186
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


CONFIGS = {  # Submission: Feature drop-list
    "01_baseline_linear": [],
    "02_pruned_linear": ["x1", "x5", "x8", "x9", "x11"],
}

TAG = sys.argv[1] if len(sys.argv) > 1 else list(CONFIGS)[-1]
DROP = CONFIGS[TAG]

HERE = Path(__file__).parent
STEM = "summer2026_kaggle_linear_regression_challenge"

train = pd.read_csv(HERE / "data" / f"{STEM}_train.csv")
test = pd.read_csv(HERE / "data" / f"{STEM}_test.csv")

X = train.drop(columns=["target", "Id"]).drop(columns=DROP)
y = train["target"]
Xt = test[X.columns]

model = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(X, y)

sub = pd.DataFrame({"Id": test["Id"].astype(int), "target": model.predict(Xt)})
out = HERE / "submissions" / f"{TAG}.csv"
out.parent.mkdir(exist_ok=True)
sub.to_csv(out, index=False)

print(f"wrote {out} ({len(sub)} rows)")
print(sub.head(3).to_string(index=False))
print("pred: mean %.2f std %.2f min %.2f max %.2f" % (
    sub.target.mean(), sub.target.std(), sub.target.min(), sub.target.max()))
