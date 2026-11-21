"""
Training utilities for the cashless health-insurance claim regression project.
Run from the repository root: python train_model.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cashless_claim_prediction_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
TARGET = "approved_amount"
ID_COLS = ["claim_id", "patient_id", "hospital_id"]

def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, row-level features available at claim assessment time."""
    data = raw.copy()
    if "claim_submitted_date" in data:
        dates = pd.to_datetime(data["claim_submitted_date"], errors="coerce")
        data["claim_month"] = dates.dt.month
        data["claim_quarter"] = dates.dt.quarter
        data["claim_dayofweek"] = dates.dt.dayofweek
        data = data.drop(columns=["claim_submitted_date"])
    # Normalize common gender variants without inventing a new demographic class.
    if "gender" in data:
        mapping = {"m": "Male", "male": "Male", "f": "Female", "female": "Female",
                   "o": "Other", "other": "Other"}
        data["gender"] = data["gender"].map(
            lambda x: mapping.get(str(x).strip().lower(), "Unknown") if pd.notna(x) else np.nan
        )
    # Invalid measurement values are treated as missing, not silently clipped.
    if "age" in data:
        data.loc[(data["age"] < 0) | (data["age"] > 120), "age"] = np.nan
    if "bmi" in data:
        data.loc[(data["bmi"] <= 0) | (data["bmi"] > 100), "bmi"] = np.nan
    if "copay_percent" in data:
        data.loc[(data["copay_percent"] < 0) | (data["copay_percent"] > 100), "copay_percent"] = np.nan
    # Cost mix and policy-pressure features. Safe denominators prevent infinities.
    def ratio(numerator, denominator):
        den = pd.to_numeric(data[denominator], errors="coerce").replace(0, np.nan)
        return pd.to_numeric(data[numerator], errors="coerce") / den
    for col, new_col in [("room_charges", "room_charge_share"),
                         ("doctor_fees", "doctor_fee_share"),
                         ("medicine_charges", "medicine_charge_share")]:
        if {col, "claimed_amount"}.issubset(data.columns):
            data[new_col] = ratio(col, "claimed_amount")
    if {"claimed_amount", "sum_insured"}.issubset(data.columns):
        data["claim_to_sum_insured_ratio"] = ratio("claimed_amount", "sum_insured")
    if {"claimed_amount", "remaining_sum_insured"}.issubset(data.columns):
        data["claim_to_remaining_cover_ratio"] = ratio("claimed_amount", "remaining_sum_insured")
        data["claim_excess_over_remaining_cover"] = (
            data["claimed_amount"] - data["remaining_sum_insured"]
        )
    if {"length_of_stay_days", "room_rent_limit_per_day"}.issubset(data.columns):
        data["stay_room_limit_exposure"] = (
            data["length_of_stay_days"] * data["room_rent_limit_per_day"]
        )
    # Identifiers are not used as model predictors; they can memorize entities.
    data = data.drop(columns=[c for c in ID_COLS if c in data], errors="ignore")
    return data

def build_pipeline(model):
    numeric_features = None
    categorical_features = None
    # The preprocessor is fitted only on training folds.
    from sklearn.compose import make_column_selector
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical_pipe, make_column_selector(dtype_exclude=np.number))
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])

def main():
    raw = pd.read_csv(DATA_PATH)
    if TARGET not in raw:
        raise ValueError(f"Target column {TARGET!r} is missing.")
    # Remove only exact duplicate rows. Repeated claim IDs are reported, not auto-deleted.
    raw = raw.drop_duplicates().copy()
    raw[TARGET] = pd.to_numeric(raw[TARGET], errors="coerce")
    raw = raw.dropna(subset=[TARGET])
    raw = raw[raw[TARGET] >= 0].copy()

    X = prepare_features(raw.drop(columns=[TARGET]))
    y = raw[TARGET].astype(float)

    # Chronological holdout: later-submitted claims are held out as a future-like test set.
    if "claim_submitted_date" in pd.read_csv(DATA_PATH, nrows=1).columns:
        dates = pd.to_datetime(raw["claim_submitted_date"], errors="coerce")
        order = dates.sort_values(kind="stable").index
        split_at = int(len(order) * 0.80)
        train_idx, test_idx = order[:split_at], order[split_at:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

    candidates = {
        "Ridge": Ridge(alpha=10.0),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2,
                                          random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=150,
                                                       learning_rate=0.05, max_depth=2)
    }
    rows, fitted = [], {}
    # Use training data CV for model selection; the holdout test set is not used here.
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=TimeSeriesSplit(n_splits=5),
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                     "r2": "r2"}, n_jobs=-1, error_score="raise"
        )
        rows.append({"model": name,
                     "CV_MAE_mean": -cv_scores["test_mae"].mean(),
                     "CV_RMSE_mean": -cv_scores["test_rmse"].mean(),
                     "CV_R2_mean": cv_scores["test_r2"].mean()})
    comparison = pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)
    best_name = comparison.loc[0, "model"]
    best_pipe = build_pipeline(candidates[best_name])
    best_pipe.fit(X_train, y_train)
    predictions = best_pipe.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "test_MAE": float(mean_absolute_error(y_test, predictions)),
        "test_RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_R2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count_before_encoding": int(X_train.shape[1]),
        "test_split": "chronological 80/20 by claim_submitted_date"
    }
    joblib.dump({"pipeline": best_pipe, "feature_columns": list(X.columns),
                 "model_name": best_name, "metrics": metrics},
                ARTIFACT_DIR / "cashless_claim_model.joblib")
    # Tree-based feature importance is a model-specific association, not causation.
    fitted_model = best_pipe.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": fitted_model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame({"actual": y_test, "predicted": predictions,
                  "residual": y_test - predictions}).to_csv(
                      ARTIFACT_DIR / "test_predictions.csv", index=False)
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\\nCross-validation model comparison (training data only):")
    print(comparison.to_string(index=False))
    print("\\nFinal holdout test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\\nSaved model to artifacts/cashless_claim_model.joblib")

if __name__ == "__main__":
    main()
