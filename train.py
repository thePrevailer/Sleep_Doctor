"""Fit the Sleep Doctor models and export their parameters to models.json.

This is the offline training step: it fits both Linear Regression models,
evaluates them on a held-out test split, and writes their coefficients,
intercept, feature means, and metrics to a small JSON artifact.

app.py loads that artifact instead of retraining, so the deployed app's
predictions and reported metrics stay pinned to whatever was fit and
validated here.

Run whenever the dataset or feature set changes:
    python train.py
"""

import json

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from model_config import (
    ARTIFACT_PATH,
    DATA_PATH,
    EXTENSION_CATEGORICAL,
    EXTENSION_RAW_COLUMNS,
    PRIMARY_FEATURES,
    RANDOM_STATE,
    to_bucket,
)


def fit_and_evaluate(X_train, y_train, X_test, y_test, bucket_test):
    model = LinearRegression().fit(X_train, y_train)
    pred_test = model.predict(X_test)
    pred_bucket = to_bucket(pd.Series(pred_test, index=X_test.index))
    return {
        "coefficients": pd.Series(model.coef_, index=X_train.columns).round(6).to_dict(),
        "intercept": round(float(model.intercept_), 6),
        "means": X_train.mean().round(6).to_dict(),
        "test_r2": round(float(r2_score(y_test, pred_test)), 4),
        "test_accuracy": round(float(accuracy_score(bucket_test, pred_bucket)), 4),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["quality_bucket"] = to_bucket(df["sleep_quality_score"])

    # Same 70/15/15 split as 05_model_refinements.ipynb (cascaded stratified splits,
    # same seed) so the app's reported numbers match the notebook exactly. Models are
    # fit on the 70% train slice only and scored on the untouched 15% test slice; the
    # 15% validation slice is unused here (it exists in the notebook for RF depth
    # tuning, which doesn't apply to Linear Regression) but is carved out identically
    # so train/test end up with the same rows.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["quality_bucket"]
    )
    _val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["quality_bucket"]
    )
    y_train, y_test = train_df["sleep_quality_score"], test_df["sleep_quality_score"]
    bucket_test = test_df["quality_bucket"]

    # Primary — the research question's three factors (activity = steps + exercise flag).
    primary_artifact = fit_and_evaluate(
        train_df[PRIMARY_FEATURES], y_train, test_df[PRIMARY_FEATURES], y_test, bucket_test
    )
    primary_artifact["features"] = PRIMARY_FEATURES

    # Extension — full lifestyle/context, one-hot encoded to match 05_model_refinements.ipynb.
    X_all = pd.get_dummies(df[EXTENSION_RAW_COLUMNS], columns=EXTENSION_CATEGORICAL, drop_first=True)
    extension_artifact = fit_and_evaluate(
        X_all.loc[train_df.index], y_train, X_all.loc[test_df.index], y_test, bucket_test
    )
    extension_artifact["columns"] = list(X_all.columns)

    artifact = {
        "random_state": RANDOM_STATE,
        "trained_on_rows": len(df),
        "primary": primary_artifact,
        "extension": extension_artifact,
    }
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Wrote {ARTIFACT_PATH}")
    print(f"  primary   R2={primary_artifact['test_r2']}  bucket accuracy={primary_artifact['test_accuracy']}")
    print(f"  extension R2={extension_artifact['test_r2']}  bucket accuracy={extension_artifact['test_accuracy']}")


if __name__ == "__main__":
    main()
