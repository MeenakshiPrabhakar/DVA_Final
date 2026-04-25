"""
Evaluate predict_rating model quality with cross-validation and baselines.

This script reuses the same features and model settings from predict_rating.py,
then reports:
  - Cross-validation fold metrics
  - Overall out-of-fold metrics
  - Baseline performance (global mean and type-mean)
  - Error breakdown by property type

Outputs are saved to model_output/ by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

from predict_rating import (
    FEATURE_COLS,
    INPUT_CSV,
    RATING_COLUMN,
    RANDOM_STATE,
    TYPE_COLUMN,
    load_data,
    rating_to_score_0_100,
)


def build_model(random_state: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        verbosity=0,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 1, 5)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "exact_rounded_accuracy": float(np.mean(np.round(y_pred) == np.round(y_true))),
        "within_0_5_star": float(np.mean(np.abs(y_pred - y_true) <= 0.5)),
        "within_1_star": float(np.mean(np.abs(y_pred - y_true) <= 1.0)),
    }


def evaluate(
    df: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_features = FEATURE_COLS + ["type_hotel"]
    X = df[model_features].values
    y = df[RATING_COLUMN].values
    prop_type = df[TYPE_COLUMN].astype(str).values

    oof_model = np.full(len(df), np.nan, dtype=float)
    oof_global_mean = np.full(len(df), np.nan, dtype=float)
    oof_type_mean = np.full(len(df), np.nan, dtype=float)

    fold_rows: list[dict[str, float | int | str]] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        type_train, type_test = prop_type[train_idx], prop_type[test_idx]

        model = build_model(random_state=random_state + fold_num)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        pred_model = np.clip(model.predict(X_test), 1, 5)
        oof_model[test_idx] = pred_model

        global_mean = float(np.mean(y_train))
        pred_global = np.full(len(test_idx), global_mean, dtype=float)
        oof_global_mean[test_idx] = pred_global

        type_means = pd.DataFrame(
            {TYPE_COLUMN: type_train, "rating": y_train}
        ).groupby(TYPE_COLUMN)["rating"].mean()
        pred_type = np.array(
            [float(type_means.get(t, global_mean)) for t in type_test], dtype=float
        )
        oof_type_mean[test_idx] = pred_type

        model_m = regression_metrics(y_test, pred_model)
        global_m = regression_metrics(y_test, pred_global)
        type_m = regression_metrics(y_test, pred_type)

        fold_rows.append({"fold": fold_num, "predictor": "xgboost", **model_m})
        fold_rows.append({"fold": fold_num, "predictor": "baseline_global_mean", **global_m})
        fold_rows.append({"fold": fold_num, "predictor": "baseline_type_mean", **type_m})

    fold_metrics = pd.DataFrame(fold_rows)

    summary_rows = []
    for name, preds in [
        ("xgboost", oof_model),
        ("baseline_global_mean", oof_global_mean),
        ("baseline_type_mean", oof_type_mean),
    ]:
        m = regression_metrics(y, preds)
        summary_rows.append(
            {
                "predictor": name,
                **m,
                "mean_predicted_score_0_100": float(
                    rating_to_score_0_100(pd.Series(preds)).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    oof = pd.DataFrame(
        {
            TYPE_COLUMN: prop_type,
            "actual_rating": y,
            "predicted_rating_xgboost": oof_model,
            "predicted_rating_baseline_global_mean": oof_global_mean,
            "predicted_rating_baseline_type_mean": oof_type_mean,
        }
    )
    oof["actual_score_0_100"] = rating_to_score_0_100(oof["actual_rating"]).round(2)
    oof["predicted_score_0_100_xgboost"] = rating_to_score_0_100(
        oof["predicted_rating_xgboost"]
    ).round(2)
    oof["residual_xgboost"] = (
        oof["predicted_rating_xgboost"] - oof["actual_rating"]
    ).round(4)

    by_type_rows = []
    for t, g in oof.groupby(TYPE_COLUMN):
        m = regression_metrics(g["actual_rating"].values, g["predicted_rating_xgboost"].values)
        by_type_rows.append({"type": t, "row_count": int(len(g)), **m})
    by_type = pd.DataFrame(by_type_rows).sort_values("mae")

    return fold_metrics, summary, by_type, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        default="model_output",
        help="Directory where evaluation outputs are written.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for fold splitting.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction (0,1] of rows for faster smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if args.sample_frac is not None and not (0 < args.sample_frac <= 1):
        raise ValueError("--sample-frac must be in (0, 1]")

    print(f"Loading {args.input}...")
    df = load_data(args.input)

    if args.sample_frac is not None and args.sample_frac < 1:
        df = df.sample(frac=args.sample_frac, random_state=args.random_state).reset_index(
            drop=True
        )
        print(f"Using sampled subset: {len(df):,} rows.")

    print(f"Running {args.folds}-fold cross-validation...")
    fold_metrics, summary, by_type, oof = evaluate(df, args.folds, args.random_state)

    fold_metrics_path = output_dir / "evaluation_cv_fold_metrics.csv"
    summary_path = output_dir / "evaluation_cv_summary.csv"
    by_type_path = output_dir / "evaluation_by_type.csv"
    oof_path = output_dir / "evaluation_oof_predictions.csv"

    fold_metrics.round(4).to_csv(fold_metrics_path, index=False)
    summary.round(4).to_csv(summary_path, index=False)
    by_type.round(4).to_csv(by_type_path, index=False)
    oof.round(4).to_csv(oof_path, index=False)

    print("\nOverall OOF metrics:")
    print(summary.round(4).to_string(index=False))
    print("\nSaved:")
    print(f"  {fold_metrics_path}")
    print(f"  {summary_path}")
    print(f"  {by_type_path}")
    print(f"  {oof_path}")


if __name__ == "__main__":
    main()
