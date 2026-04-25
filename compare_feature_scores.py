"""
Compare hotel vs Airbnb using model-based feature scores (no rating target).

This script does three things:
1) Unsupervised scoring: ranks features by PCA loading contribution.
2) Supervised scoring: ranks features by XGBoost importance for classifying
   property type (hotel vs airbnb).
3) Produces side-by-side comparison tables using per-feature and composite scores.

Outputs (in OUTPUT_DIR):
- feature_scores.csv
- top_features.csv
- feature_comparison_by_type.csv
- overall_score_by_type.csv
- group_winner.csv (only if GROUP_COLUMN exists)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# Configuration
# =========================
HOTEL_CSV = "reviews_scored_hotel.csv"
AIRBNB_CSVS = [
    "reviews_scored_newyork.csv",
    "reviews_scored_losangeles.csv",
]

OUTPUT_DIR = "model_output"
TOP_N = 5
RANDOM_STATE = 42
MAX_ROWS_PER_SOURCE = 150000

# Set this to "postal_code" / "zipcode" / "neighbourhood" once present
GROUP_COLUMN: str | None = None
MIN_GROUP_COUNT = 30


# =========================
# Helpers
# =========================

def _pick_feature_columns(df: pd.DataFrame) -> list[str]:
    """Use normalized frequency features only (skip *_raw columns)."""
    cols = [c for c in df.columns if c.startswith("freq_") and not c.endswith("_raw")]
    return sorted(cols)


def _read_with_label(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["property_type"] = label
    return df


def _normalize_0_100(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if np.isclose(lo, hi):
        return np.full_like(arr, 50.0)
    return (arr - lo) / (hi - lo) * 100.0


def unsupervised_feature_scores(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Rank features by weighted absolute PCA loadings.
    This is unsupervised (no labels used).
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_components = min(5, Xs.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(Xs)

    loadings = np.abs(pca.components_)  # shape: (n_components, n_features)
    weights = pca.explained_variance_ratio_.reshape(-1, 1)
    raw_scores = (loadings * weights).sum(axis=0)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "unsupervised_score": _normalize_0_100(raw_scores).round(4),
        }
    )


def supervised_feature_scores(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rank features using XGBoost classifier importance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_test": int(len(y_test)),
    }

    raw_scores = model.feature_importances_
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "supervised_score": _normalize_0_100(raw_scores).round(4),
        }
    )
    return out, metrics


def main() -> None:
    print("Loading scored datasets...")
    hotel = _read_with_label(HOTEL_CSV, "hotel")

    airbnb_parts = []
    for p in AIRBNB_CSVS:
        path = Path(p)
        if path.exists():
            airbnb_parts.append(_read_with_label(p, "airbnb"))
    if not airbnb_parts:
        raise FileNotFoundError("No Airbnb scored CSVs found. Check AIRBNB_CSVS config.")

    airbnb = pd.concat(airbnb_parts, ignore_index=True)

    hotel_features = set(_pick_feature_columns(hotel))
    airbnb_features = set(_pick_feature_columns(airbnb))
    feature_cols = sorted(hotel_features & airbnb_features)

    if len(feature_cols) < 2:
        raise ValueError(
            "Need at least 2 shared freq_ feature columns across hotel and Airbnb datasets."
        )

    keep_cols = feature_cols + ["property_type"]
    if GROUP_COLUMN and GROUP_COLUMN in hotel.columns and GROUP_COLUMN in airbnb.columns:
        keep_cols.append(GROUP_COLUMN)

    hotel = hotel[keep_cols].copy()
    airbnb = airbnb[keep_cols].copy()

    for col in feature_cols:
        hotel[col] = pd.to_numeric(hotel[col], errors="coerce")
        airbnb[col] = pd.to_numeric(airbnb[col], errors="coerce")

    hotel = hotel.dropna(subset=feature_cols)
    airbnb = airbnb.dropna(subset=feature_cols)

    if len(hotel) > MAX_ROWS_PER_SOURCE:
        hotel = hotel.sample(n=MAX_ROWS_PER_SOURCE, random_state=RANDOM_STATE)
    if len(airbnb) > MAX_ROWS_PER_SOURCE:
        airbnb = airbnb.sample(n=MAX_ROWS_PER_SOURCE, random_state=RANDOM_STATE)

    df = pd.concat([hotel, airbnb], ignore_index=True)
    print(f"Using {len(df):,} rows and {len(feature_cols)} shared features.")

    X = df[feature_cols].values
    y = (df["property_type"] == "airbnb").astype(int).values

    print("Scoring features (unsupervised + supervised)...")
    unsup = unsupervised_feature_scores(X, feature_cols)
    sup, clf_metrics = supervised_feature_scores(X, y, feature_cols)

    scores = unsup.merge(sup, on="feature", how="inner")
    scores["combined_score"] = (
        0.5 * scores["unsupervised_score"] + 0.5 * scores["supervised_score"]
    ).round(4)
    scores = scores.sort_values("combined_score", ascending=False).reset_index(drop=True)

    top_features = scores.head(TOP_N).copy()

    print("Top features:")
    print(top_features[["feature", "combined_score"]].to_string(index=False))

    scaler = StandardScaler()
    z = scaler.fit_transform(df[feature_cols].values)

    weight_lookup = {
        row.feature: row.combined_score for row in scores.itertuples(index=False)
    }
    raw_weights = np.array([weight_lookup[f] for f in feature_cols], dtype=float)
    raw_weights = np.maximum(raw_weights, 0)
    if np.isclose(raw_weights.sum(), 0):
        raw_weights = np.ones_like(raw_weights)
    weights = raw_weights / raw_weights.sum()

    df["composite_score"] = np.dot(z, weights)

    by_type = (
        df.groupby("property_type")[feature_cols + ["composite_score"]]
        .mean()
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    feature_compare = pd.DataFrame({"feature": feature_cols})
    hotel_means = df[df["property_type"] == "hotel"][feature_cols].mean()
    airbnb_means = df[df["property_type"] == "airbnb"][feature_cols].mean()
    feature_compare["hotel_mean"] = feature_compare["feature"].map(hotel_means)
    feature_compare["airbnb_mean"] = feature_compare["feature"].map(airbnb_means)
    feature_compare["delta_airbnb_minus_hotel"] = (
        feature_compare["airbnb_mean"] - feature_compare["hotel_mean"]
    ).round(4)

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    scores.to_csv(out / "feature_scores.csv", index=False)
    top_features.to_csv(out / "top_features.csv", index=False)
    feature_compare.to_csv(out / "feature_comparison_by_type.csv", index=False)
    by_type.to_csv(out / "overall_score_by_type.csv", index=False)

    if GROUP_COLUMN and GROUP_COLUMN in df.columns:
        group_stats = (
            df.groupby([GROUP_COLUMN, "property_type"])
            .agg(
                mean_composite_score=("composite_score", "mean"),
                n=("composite_score", "count"),
            )
            .reset_index()
        )

        pivot = group_stats.pivot(index=GROUP_COLUMN, columns="property_type", values="mean_composite_score")
        counts = group_stats.pivot(index=GROUP_COLUMN, columns="property_type", values="n").fillna(0)

        if "airbnb" in pivot.columns and "hotel" in pivot.columns:
            group_winner = pivot[["airbnb", "hotel"]].copy()
            group_winner.columns = ["airbnb_score", "hotel_score"]
            group_winner["delta_airbnb_minus_hotel"] = (
                group_winner["airbnb_score"] - group_winner["hotel_score"]
            )
            group_winner["airbnb_n"] = counts.get("airbnb", 0)
            group_winner["hotel_n"] = counts.get("hotel", 0)
            group_winner = group_winner.reset_index()

            group_winner = group_winner[
                (group_winner["airbnb_n"] >= MIN_GROUP_COUNT)
                & (group_winner["hotel_n"] >= MIN_GROUP_COUNT)
            ].copy()

            group_winner["winner"] = np.where(
                group_winner["delta_airbnb_minus_hotel"] > 0,
                "airbnb",
                "hotel",
            )
            group_winner.to_csv(out / "group_winner.csv", index=False)

    print("\nClassifier quality (for property type separation):")
    print(f"  AUC:      {clf_metrics['auc']:.4f}")
    print(f"  Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"  N test:   {clf_metrics['n_test']}")

    print(f"\nSaved outputs to {out}/")
    print("  feature_scores.csv")
    print("  top_features.csv")
    print("  feature_comparison_by_type.csv")
    print("  overall_score_by_type.csv")
    if GROUP_COLUMN:
        print("  group_winner.csv (if group column exists on both datasets)")


if __name__ == "__main__":
    main()
