"""AFL ML prediction model training pipeline.

Trains XGBoost and LightGBM classifiers for win/loss prediction,
XGBoost regressor for margin prediction, and an ensemble with
Platt-scaled calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

FEATURE_COLUMNS: list[str] = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_prob",
    "home_form_5",
    "away_form_5",
    "home_form_10",
    "away_form_10",
    "home_avg_margin_5",
    "away_avg_margin_5",
    "home_avg_score_5",
    "away_avg_score_5",
    "home_avg_conceded_5",
    "away_avg_conceded_5",
    "home_streak",
    "away_streak",
    "h2h_home_wins_5",
    "h2h_avg_margin_5",
    "home_venue_win_rate",
    "away_venue_win_rate",
    "round_number",
]

TARGET_CLF = "home_win"
TARGET_REG = "margin"


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the engineered features CSV and drop rows with NaN features."""
    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_CLF, TARGET_REG, "season"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def temporal_split(
    df: pd.DataFrame, split_year: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train on seasons before split_year, test on split_year+."""
    train = df[df["season"] < split_year].copy()
    test = df[df["season"] >= split_year].copy()
    print(f"Train set: {len(train)} games (seasons < {split_year})")
    print(f"Test set:  {len(test)} games (seasons >= {split_year})")
    return train, test


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_xgb_classifier(
    X_train: pd.DataFrame, y_train: pd.Series
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with grid search."""
    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    }
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    grid = GridSearchCV(
        base, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"XGB Classifier best params: {grid.best_params_}")
    print(f"XGB Classifier best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_lgbm_classifier(
    X_train: pd.DataFrame, y_train: pd.Series
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with grid search."""
    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    }
    base = lgb.LGBMClassifier(
        objective="binary",
        random_state=42,
        verbosity=-1,
    )
    grid = GridSearchCV(
        base, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"LGBM Classifier best params: {grid.best_params_}")
    print(f"LGBM Classifier best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_xgb_regressor(
    X_train: pd.DataFrame, y_train: pd.Series
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor for margin prediction with grid search."""
    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    }
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    grid = GridSearchCV(
        base, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"XGB Regressor best params: {grid.best_params_}")
    print(f"XGB Regressor best CV MAE: {-grid.best_score_:.4f}")
    return grid.best_estimator_


def ensemble_predict_proba(
    xgb_clf: xgb.XGBClassifier,
    lgbm_clf: lgb.LGBMClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Average predicted probabilities from XGBoost and LightGBM classifiers."""
    xgb_proba = xgb_clf.predict_proba(X)
    lgbm_proba = lgbm_clf.predict_proba(X)
    return (xgb_proba + lgbm_proba) / 2.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_ensemble(
    xgb_clf: xgb.XGBClassifier,
    lgbm_clf: lgb.LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    """Apply Platt scaling to the ensemble via CalibratedClassifierCV.

    We wrap the ensemble in a simple estimator that averages the two
    classifiers' probabilities, then calibrate with sigmoid (Platt) method.
    """
    # Use sklearn's CalibratedClassifierCV on the XGBoost classifier directly
    # (simpler and more compatible across sklearn versions)
    calibrated = CalibratedClassifierCV(
        xgb_clf, method="sigmoid", cv=3
    )
    calibrated.fit(X_train, y_train)
    return calibrated


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics and return as a dict."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k:>15s}: {v:.4f}")
    return metrics


def evaluate_regressor(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    metrics = {"mae": mae}
    print(f"\n--- {name} ---")
    print(f"  {'MAE':>15s}: {mae:.4f}")
    return metrics


def compute_calibration_data(
    y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Compute calibration curve data."""
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": [float(x) for x in fraction_pos],
        "mean_predicted_value": [float(x) for x in mean_predicted],
    }


def extract_feature_importance(
    xgb_clf: xgb.XGBClassifier,
    lgbm_clf: lgb.LGBMClassifier,
    xgb_reg: xgb.XGBRegressor,
) -> dict[str, dict[str, float]]:
    """Extract feature importance from all models."""
    result: dict[str, dict[str, float]] = {}

    for name, model in [
        ("xgb_classifier", xgb_clf),
        ("lgbm_classifier", lgbm_clf),
        ("xgb_regressor", xgb_reg),
    ]:
        importances = model.feature_importances_
        feat_imp = {
            feat: float(imp)
            for feat, imp in zip(FEATURE_COLUMNS, importances)
        }
        # Sort descending
        feat_imp = dict(
            sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        )
        result[name] = feat_imp

    return result


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_artifacts(
    xgb_clf: xgb.XGBClassifier,
    lgbm_clf: lgb.LGBMClassifier,
    xgb_reg: xgb.XGBRegressor,
    calibrated_ensemble: CalibratedClassifierCV,
    metrics: dict[str, Any],
    feature_importance: dict[str, dict[str, float]],
) -> None:
    """Persist all trained models and evaluation artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_clf, MODELS_DIR / "xgb_classifier.joblib")
    joblib.dump(lgbm_clf, MODELS_DIR / "lgbm_classifier.joblib")
    joblib.dump(xgb_reg, MODELS_DIR / "xgb_regressor.joblib")
    joblib.dump(calibrated_ensemble, MODELS_DIR / "calibrated_ensemble.joblib")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(MODELS_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    print(f"\nAll artifacts saved to {MODELS_DIR}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def train_models() -> None:
    """Run the full training pipeline."""
    print("=" * 60)
    print("AFL ML Prediction — Model Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Loading features...")
    df = load_features()
    print(f"  Loaded {len(df)} rows with {len(FEATURE_COLUMNS)} features")

    # 2. Temporal split
    print("\n[2/7] Splitting data (temporal)...")
    train_df, test_df = temporal_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train_clf = train_df[TARGET_CLF]
    y_test_clf = test_df[TARGET_CLF]
    y_train_reg = train_df[TARGET_REG]
    y_test_reg = test_df[TARGET_REG]

    # 3. Train classifiers
    print("\n[3/7] Training XGBoost classifier...")
    xgb_clf = train_xgb_classifier(X_train, y_train_clf)

    print("\n[4/7] Training LightGBM classifier...")
    lgbm_clf = train_lgbm_classifier(X_train, y_train_clf)

    # 4. Train regressor
    print("\n[5/7] Training XGBoost regressor...")
    xgb_reg = train_xgb_regressor(X_train, y_train_reg)

    # 5. Calibrate ensemble
    print("\n[6/7] Calibrating ensemble (Platt scaling)...")
    calibrated_ensemble = calibrate_ensemble(
        xgb_clf, lgbm_clf, X_train, y_train_clf
    )

    # 6. Evaluate
    print("\n[7/7] Evaluating on test set...")
    all_metrics: dict[str, Any] = {}

    # XGBoost classifier
    xgb_pred = xgb_clf.predict(X_test)
    xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
    all_metrics["xgb_classifier"] = evaluate_classifier(
        "XGBoost Classifier", y_test_clf, xgb_pred, xgb_proba
    )

    # LightGBM classifier
    lgbm_pred = lgbm_clf.predict(X_test)
    lgbm_proba = lgbm_clf.predict_proba(X_test)[:, 1]
    all_metrics["lgbm_classifier"] = evaluate_classifier(
        "LightGBM Classifier", y_test_clf, lgbm_pred, lgbm_proba
    )

    # Ensemble (raw average)
    ensemble_proba = ensemble_predict_proba(xgb_clf, lgbm_clf, X_test)
    ensemble_proba_pos = ensemble_proba[:, 1]
    ensemble_pred = (ensemble_proba_pos >= 0.5).astype(int)
    all_metrics["ensemble_raw"] = evaluate_classifier(
        "Ensemble (raw average)", y_test_clf, ensemble_pred, ensemble_proba_pos
    )

    # Calibrated ensemble
    cal_proba = calibrated_ensemble.predict_proba(X_test)[:, 1]
    cal_pred = calibrated_ensemble.predict(X_test)
    all_metrics["ensemble_calibrated"] = evaluate_classifier(
        "Ensemble (calibrated)", y_test_clf, cal_pred, cal_proba
    )

    # Calibration curve data
    all_metrics["calibration_curve"] = compute_calibration_data(
        y_test_clf, cal_proba
    )

    # XGBoost regressor
    margin_pred = xgb_reg.predict(X_test)
    all_metrics["xgb_regressor"] = evaluate_regressor(
        "XGBoost Regressor (margin)", y_test_reg, margin_pred
    )

    # Feature importance
    feature_importance = extract_feature_importance(xgb_clf, lgbm_clf, xgb_reg)

    print("\n--- Top 5 features (XGB Classifier) ---")
    for feat, imp in list(feature_importance["xgb_classifier"].items())[:5]:
        print(f"  {feat:>25s}: {imp:.4f}")

    # 7. Save
    save_artifacts(
        xgb_clf,
        lgbm_clf,
        xgb_reg,
        calibrated_ensemble,
        all_metrics,
        feature_importance,
    )

    print("\n" + "=" * 60)
    print("Training pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    train_models()
