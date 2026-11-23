"""
Member 4 - Machine learning indicator: probability that a booking will be cancelled.

What it does
1. Loads the cleaned dataset (Data/clean/hotel_bookings_clean.csv).
2. Compares two models: Logistic Regression (simple baseline) and Gradient Boosting.
3. Keeps the model with the best ROC AUC and saves it to models/cancel_model.joblib.
4. Adds two new indicator columns for the dashboard:
     cancel_probability  (0 to 1)
     risk_level          (Low / Medium / High)
   and saves them to Data/clean/hotel_bookings_scored.csv.
5. Saves model results to reports/model_metrics.json (used by the report and the dashboard).

Run from anywhere:  python Scripts/ml_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- Paths (work whatever folder you run the script from) ----------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ("Data" if (ROOT / "Data").exists() else "data")
CLEAN = DATA / "clean" / "hotel_bookings_clean.csv"
SCORED = DATA / "clean" / "hotel_bookings_scored.csv"
MODEL_PATH = ROOT / "models" / "cancel_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"

# ---------- Features ----------
TARGET = "is_canceled"

# Only information the hotel knows at the moment of booking.
CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel", "deposit_type",
               "customer_type", "reserved_room_type", "arrival_date_month"]
NUMERIC = ["lead_time", "total_nights", "total_guests", "adr", "is_repeated_guest",
           "previous_cancellations", "previous_bookings_not_canceled", "days_in_waiting_list",
           "required_car_parking_spaces", "total_of_special_requests"]

# Deliberately NOT used (data leakage = information only known after the outcome):
#   reservation_status, reservation_status_date -> directly say "Canceled" / "Check-Out"
#   assigned_room_type                          -> decided at check-in
#   booking_changes                             -> can change until arrival
#   country                                     -> often filled in only at check-in for this dataset

RISK_BINS = [0, 0.3, 0.6, 1]
RISK_LABELS = ["Low", "Medium", "High"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the two totals the model needs (kept here so the ML step does not depend on other scripts)."""
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    return df


def make_pipeline(model, scale_numbers: bool) -> Pipeline:
    """Preprocessing (text -> 0/1 columns, optional scaling) followed by the model."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler() if scale_numbers else "passthrough", NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def candidate_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=2000), scale_numbers=True),
        "Gradient Boosting": make_pipeline(
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42),
            scale_numbers=False),
    }


def evaluate(pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
    }


def train(df: pd.DataFrame):
    """Compare the models, keep the best one, return (fitted model, metrics dict)."""
    df = add_features(df)
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results = {}
    fitted = {}
    for name, pipe in candidate_models().items():
        pipe.fit(X_tr, y_tr)
        results[name] = evaluate(pipe, X_te, y_te)
        fitted[name] = pipe
        print(f"{name:20s} {results[name]}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = fitted[best_name]
    print(f"\nBest model: {best_name} (ROC AUC {results[best_name]['roc_auc']})")

    # Which inputs matter most? (shuffle one column at a time and measure how much AUC drops)
    sample = X_te.sample(min(3000, len(X_te)), random_state=42)
    imp = permutation_importance(best, sample, y_te.loc[sample.index], scoring="roc_auc",
                                 n_repeats=3, random_state=42)
    importance = (pd.Series(imp.importances_mean, index=sample.columns)
                    .sort_values(ascending=False).round(4))

    metrics = {
        "best_model": best_name,
        "comparison": results,
        "feature_importance": importance.head(10).to_dict(),
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "cancellation_rate": round(float(y.mean()), 3),
    }
    return best_name, metrics


def score(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add cancel_probability and risk_level to any cleaned DataFrame (used for new batches)."""
    out = df.copy()
    out["cancel_probability"] = model.predict_proba(add_features(df)[CATEGORICAL + NUMERIC])[:, 1].round(4)
    out["risk_level"] = pd.cut(out["cancel_probability"], RISK_BINS, labels=RISK_LABELS,
                               include_lowest=True)
    return out


def main():
    data = pd.read_csv(CLEAN)
    best_name, metrics = train(data)

    # Honest probabilities for the dashboard: each booking is scored by a model that never saw it
    # (5-fold cross-validation). Scoring the training rows directly would look better than reality.
    feats = add_features(data)
    X, y = feats[CATEGORICAL + NUMERIC], feats[TARGET]
    proba = cross_val_predict(candidate_models()[best_name], X, y,
                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                              method="predict_proba")[:, 1]
    scored = data.copy()
    scored["cancel_probability"] = proba.round(4)
    scored["risk_level"] = pd.cut(scored["cancel_probability"], RISK_BINS, labels=RISK_LABELS,
                                  include_lowest=True)

    # Final model trained on all rows, saved for scoring future batches
    final_model = candidate_models()[best_name].fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH, compress=3)
    scored.to_csv(SCORED, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nRisk levels:\n{scored['risk_level'].value_counts().to_string()}")
    print(f"\nSaved model    -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved scores   -> {SCORED.relative_to(ROOT)}")
    print(f"Saved metrics  -> {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
