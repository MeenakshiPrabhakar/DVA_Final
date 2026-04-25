"""
<<<<<<< HEAD
XGBoost rating prediction — Option A.

This script runs AFTER score_reviews.py.  It takes reviews_scored.csv
and trains an XGBoost model to predict overall rating (1–5) from the
five frequency scores:
    freq_location, freq_comfort, freq_airport,
    freq_cleanliness, freq_service

Outputs
-------
  model_output/
    predictions.csv          — every review with predicted vs actual rating
                               and the residual (predicted - actual)
    feature_importance.csv   — which of the 5 scores matter most
    metrics.csv              — MAE, RMSE, accuracy summary
    residual_summary.csv     — mean residual grouped by ZIP / neighbourhood
                               (use this to colour the choropleth map)

Edit the configuration section below before running.

Usage
-----
    pip install xgboost scikit-learn pandas matplotlib
    python predict_rating.py
=======
Predict comparable property ratings from merged feature scores.

Input:
  merged_features.csv

Expected columns in merged_features.csv:
  rating, type, freq_host, freq_amenities, freq_space, freq_transport, freq_location

Outputs (model_output/):
  metrics.csv                  - holdout performance
  feature_importance.csv       - model feature importance
  predictions_all.csv          - all rows with predicted rating and normalized score
  residual_summary_by_type.csv - residual summary for Airbnb vs Hotel
>>>>>>> ae2e806 (latest added files)
"""

from __future__ import annotations

<<<<<<< HEAD
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


# =========================
# Configuration
# =========================
INPUT_CSV   = "bertopic_output/reviews_scored.csv"
OUTPUT_DIR  = "model_output"

# Column that holds the overall 1–5 star rating.
#   Hotel dataset (CMU)  → "ratings"  (needs parsing, see below)
#   Airbnb dataset       → "review_scores_rating"  (may be 0–100 scale, see note)
RATING_COLUMN = "overall_rating"

# If your dataset stores ratings as JSON (CMU hotel), set this to True
# and the script will parse the "overall" key out of the ratings field.
PARSE_RATING_FROM_JSON = False
RAW_RATING_COLUMN      = "ratings"   # only used when PARSE_RATING_FROM_JSON = True

# Optional: column containing ZIP code or neighbourhood for residual map.
# Set to None if your data does not have one.
GEO_COLUMN = "neighbourhood"   # e.g. "neighbourhood", "zipcode", "postal_code"

RANDOM_STATE = 42
TEST_SIZE    = 0.2

FEATURE_COLS = [
    "freq_location",
    "freq_comfort",
    "freq_airport",
    "freq_cleanliness",
    "freq_service",
]


# =========================
# Helpers
# =========================

def parse_overall_rating(ratings_str: str) -> float | None:
    """
    Extract the overall rating from a JSON-like ratings string.
    Used for the CMU hotel dataset where ratings look like:
        {'service': 4.0, 'cleanliness': 5.0, 'overall': 4.0, ...}
    """
    import json
    try:
        r = json.loads(str(ratings_str).replace("'", '"'))
        val = r.get("overall") or r.get("Overall")
        return float(val) if val is not None else None
    except Exception:
        return None


def normalise_rating(series: pd.Series) -> pd.Series:
    """
    Normalise ratings to a 1–5 scale.
    Airbnb stores review_scores_rating as 0–100 (older data) or 0–5.
    Anything above 10 is assumed to be 0–100 and rescaled.
    """
    if series.max() > 10:
        return (series / 20).clip(1, 5)
    return series.clip(1, 5)


# =========================
# Main
# =========================

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # ── Rating column ──────────────────────────────────────────
    if PARSE_RATING_FROM_JSON:
        if RAW_RATING_COLUMN not in df.columns:
            raise ValueError(
                f"PARSE_RATING_FROM_JSON is True but '{RAW_RATING_COLUMN}' "
                f"column not found. Available: {list(df.columns)}"
            )
        df[RATING_COLUMN] = df[RAW_RATING_COLUMN].apply(parse_overall_rating)
    else:
        if RATING_COLUMN not in df.columns:
            raise ValueError(
                f"Rating column '{RATING_COLUMN}' not found. "
                f"Available columns: {list(df.columns)}\n"
                "Set RATING_COLUMN or PARSE_RATING_FROM_JSON in the config."
            )

    df[RATING_COLUMN] = pd.to_numeric(df[RATING_COLUMN], errors="coerce")
    df[RATING_COLUMN] = normalise_rating(df[RATING_COLUMN])

    # ── Feature columns ────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}\n"
            "Run score_reviews.py first."
        )

    # Drop rows where rating or any feature is null
    required = FEATURE_COLS + [RATING_COLUMN]
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows with missing rating or feature values.")

    print(f"Using {len(df):,} reviews for modelling.")
    return df.reset_index(drop=True)


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[RATING_COLUMN].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
=======
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


INPUT_CSV = "merged_features.csv"
OUTPUT_DIR = Path("model_output")

RATING_COLUMN = "rating"
TYPE_COLUMN = "type"

FEATURE_COLS = [
    "freq_host",
    "freq_amenities",
    "freq_space",
    "freq_transport",
    "freq_location",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def normalize_rating_to_1_5(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if s.max() > 10:
        return (s / 20).clip(1, 5)
    return s.clip(1, 5)


def rating_to_score_0_100(series: pd.Series) -> pd.Series:
    return (((series - 1.0) / 4.0) * 100.0).clip(0, 100)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = [RATING_COLUMN, TYPE_COLUMN] + FEATURE_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[RATING_COLUMN] = normalize_rating_to_1_5(df[RATING_COLUMN])
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use source type as a model feature so calibration can differ by source.
    df["type_hotel"] = (df[TYPE_COLUMN].astype(str).str.upper() == "H").astype(int)

    before = len(df)
    df = df.dropna(subset=[RATING_COLUMN] + FEATURE_COLS)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows with missing rating/features.")

    print(f"Using {len(df):,} rows for modelling.")
    return df.reset_index(drop=True)


def train_and_evaluate(df: pd.DataFrame):
    model_features = FEATURE_COLS + ["type_hotel"]
    X = df[model_features].values
    y = df[RATING_COLUMN].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
>>>>>>> ae2e806 (latest added files)
        random_state=RANDOM_STATE,
        verbosity=0,
    )

<<<<<<< HEAD
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    return model, y_pred, y_test, idx_test


def evaluate(y_pred: np.ndarray, y_test: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # "Correct" = predicted rounded rating matches actual rounded rating
    accuracy = np.mean(np.round(y_pred) == np.round(y_test))

    # Within-1 accuracy (predicted within 1 star of actual)
    within_1 = np.mean(np.abs(y_pred - y_test) <= 1.0)

    metrics = {
        "MAE":             round(mae, 4),
        "RMSE":            round(rmse, 4),
        "Exact accuracy":  round(accuracy, 4),
        "Within-1 star":   round(within_1, 4),
        "N test reviews":  len(y_test),
    }

    print("\nModel performance:")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")

    return metrics
=======
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = np.clip(model.predict(X_test), 1, 5)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    exact_acc = np.mean(np.round(y_pred) == np.round(y_test))
    within_half = np.mean(np.abs(y_pred - y_test) <= 0.5)
    within_one = np.mean(np.abs(y_pred - y_test) <= 1.0)

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "exact_rounded_accuracy": round(float(exact_acc), 4),
        "within_0_5_star": round(float(within_half), 4),
        "within_1_star": round(float(within_one), 4),
        "n_test": int(len(y_test)),
    }

    return model, metrics, idx_test, y_test, y_pred
>>>>>>> ae2e806 (latest added files)


def save_outputs(
    df: pd.DataFrame,
<<<<<<< HEAD
    model,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    idx_test: np.ndarray,
    metrics: dict,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    # ── predictions.csv ───────────────────────────────────────
    pred_df = df.loc[idx_test].copy()
    pred_df["predicted_rating"] = y_pred.round(2)
    pred_df["actual_rating"]    = y_test
    pred_df["residual"]         = (y_pred - y_test).round(4)
    # positive residual = overpredicted (model > actual = underperformed)
    # negative residual = underpredicted (model < actual = hidden gem)
    pred_df["interpretation"] = pred_df["residual"].apply(
        lambda r: "underperformed" if r > 0.5
        else ("hidden_gem" if r < -0.5 else "as_expected")
    )
    pred_df.to_csv(out / "predictions.csv", index=False)

    # ── feature_importance.csv ────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(out / "feature_importance.csv", index=False)

    print("\nFeature importance:")
    print(importance_df.to_string(index=False))

    # ── metrics.csv ───────────────────────────────────────────
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)

    # ── residual_summary.csv (for choropleth) ─────────────────
    if GEO_COLUMN and GEO_COLUMN in pred_df.columns:
        residual_summary = (
            pred_df.groupby(GEO_COLUMN)
            .agg(
                mean_residual      = ("residual", "mean"),
                mean_actual_rating = ("actual_rating", "mean"),
                mean_pred_rating   = ("predicted_rating", "mean"),
                review_count       = ("residual", "count"),
            )
            .round(4)
            .reset_index()
        )
        residual_summary.to_csv(out / "residual_summary.csv", index=False)
        print(f"\nResidual summary by '{GEO_COLUMN}' saved.")
        print("  Positive mean_residual = area underperforms expectations")
        print("  Negative mean_residual = area is a hidden gem")
    else:
        print(
            f"\nNote: GEO_COLUMN '{GEO_COLUMN}' not found — "
            "residual_summary.csv not created. "
            "Set GEO_COLUMN to your ZIP/neighbourhood column name."
        )

    # ── feature importance bar chart ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color="#7c3aed",
    )
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance — Rating Prediction")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved all outputs to {out}/")
    print(f"  predictions.csv")
    print(f"  feature_importance.csv + .png")
    print(f"  metrics.csv")
    if GEO_COLUMN and GEO_COLUMN in pred_df.columns:
        print(f"  residual_summary.csv  ← feed this into your choropleth")


def main() -> None:
    print("Loading scored reviews...")
    df = load_data(INPUT_CSV)

    print("\nTraining XGBoost model...")
    model, y_pred, y_test, idx_test = train_model(df)

    metrics = evaluate(y_pred, y_test)

    print("\nSaving outputs...")
    save_outputs(df, model, y_pred, y_test, idx_test, metrics, OUTPUT_DIR)
=======
    model: xgb.XGBRegressor,
    metrics: dict,
    idx_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_features = FEATURE_COLS + ["type_hotel"]

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": model_features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    scored = df.copy()
    scored["predicted_rating"] = np.clip(
        model.predict(scored[model_features].values), 1, 5
    )
    scored["predicted_rating"] = scored["predicted_rating"].round(4)
    scored["score_0_100"] = rating_to_score_0_100(scored["predicted_rating"]).round(2)
    scored["actual_score_0_100"] = rating_to_score_0_100(scored[RATING_COLUMN]).round(2)
    scored["residual"] = (scored["predicted_rating"] - scored[RATING_COLUMN]).round(4)

    scored.to_csv(OUTPUT_DIR / "predictions_all.csv", index=False)

    residual_summary = (
        scored.groupby(TYPE_COLUMN)
        .agg(
            mean_actual_rating=(RATING_COLUMN, "mean"),
            mean_predicted_rating=("predicted_rating", "mean"),
            mean_actual_score_0_100=("actual_score_0_100", "mean"),
            mean_predicted_score_0_100=("score_0_100", "mean"),
            mean_residual=("residual", "mean"),
            row_count=("residual", "count"),
        )
        .round(4)
        .reset_index()
    )
    residual_summary.to_csv(OUTPUT_DIR / "residual_summary_by_type.csv", index=False)

    holdout = df.loc[idx_test, [TYPE_COLUMN]].copy()
    holdout["actual_rating"] = y_test
    holdout["predicted_rating"] = y_pred
    holdout["actual_score_0_100"] = rating_to_score_0_100(holdout["actual_rating"]).round(2)
    holdout["predicted_score_0_100"] = rating_to_score_0_100(
        holdout["predicted_rating"]
    ).round(2)
    holdout["residual"] = (holdout["predicted_rating"] - holdout["actual_rating"]).round(4)
    holdout.to_csv(OUTPUT_DIR / "predictions_holdout.csv", index=False)

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'metrics.csv'}")
    print(f"  {OUTPUT_DIR / 'feature_importance.csv'}")
    print(f"  {OUTPUT_DIR / 'predictions_all.csv'}")
    print(f"  {OUTPUT_DIR / 'predictions_holdout.csv'}")
    print(f"  {OUTPUT_DIR / 'residual_summary_by_type.csv'}")


def main() -> None:
    print(f"Loading {INPUT_CSV}...")
    df = load_data(INPUT_CSV)

    print("Training model and evaluating on holdout split...")
    model, metrics, idx_test, y_test, y_pred = train_and_evaluate(df)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nSaving outputs...")
    save_outputs(df, model, metrics, idx_test, y_test, y_pred)
>>>>>>> ae2e806 (latest added files)

    print("\nDone.")


if __name__ == "__main__":
    main()
