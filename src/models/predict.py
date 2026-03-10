"""AFL ML prediction module.

Loads saved models and serves predictions for individual matches,
including SHAP-based feature explanations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "saved"


class ModelPredictor:
    """Loads trained models once and serves predictions."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._load_models()

    def _load_models(self) -> None:
        """Load all persisted model artifacts."""
        self.xgb_clf = joblib.load(self.models_dir / "xgb_classifier.joblib")
        self.lgbm_clf = joblib.load(self.models_dir / "lgbm_classifier.joblib")
        self.xgb_reg = joblib.load(self.models_dir / "xgb_regressor.joblib")
        self.calibrated_ensemble = joblib.load(
            self.models_dir / "calibrated_ensemble.joblib"
        )
        # Initialise SHAP explainers (tree-based, fast)
        self.xgb_explainer = shap.TreeExplainer(self.xgb_clf)
        self.lgbm_explainer = shap.TreeExplainer(self.lgbm_clf)

    def _get_shap_explanation(
        self, X: pd.DataFrame, top_n: int = 5
    ) -> list[dict[str, Any]]:
        """Return the top-N features driving the prediction via SHAP.

        Averages absolute SHAP values from both classifiers to reflect
        the ensemble's reasoning.
        """
        xgb_shap = self.xgb_explainer.shap_values(X)
        lgbm_shap = self.lgbm_explainer.shap_values(X)

        # Handle the case where shap_values returns a list (one per class)
        if isinstance(xgb_shap, list):
            xgb_shap = xgb_shap[1]  # positive class
        if isinstance(lgbm_shap, list):
            lgbm_shap = lgbm_shap[1]

        xgb_vals = np.array(xgb_shap).flatten()
        lgbm_vals = np.array(lgbm_shap).flatten()

        avg_shap = (np.abs(xgb_vals) + np.abs(lgbm_vals)) / 2.0
        avg_signed = (xgb_vals + lgbm_vals) / 2.0

        top_indices = np.argsort(avg_shap)[::-1][:top_n]

        explanations: list[dict[str, Any]] = []
        for idx in top_indices:
            feature_name = FEATURE_COLUMNS[idx]
            explanations.append(
                {
                    "feature": feature_name,
                    "value": float(X.iloc[0, idx]),
                    "shap_value": float(avg_signed[idx]),
                    "impact": "positive" if avg_signed[idx] > 0 else "negative",
                }
            )
        return explanations

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        venue: str,
        round_num: int,
        features_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate a prediction for a single match.

        Parameters
        ----------
        home_team : str
            Name of the home team.
        away_team : str
            Name of the away team.
        venue : str
            Venue name (used for context in the result, not as a model feature
            directly — venue win rates should already be in features_df).
        round_num : int
            Round number (should match ``round_number`` in features_df).
        features_df : pd.DataFrame
            A single-row DataFrame containing all ``FEATURE_COLUMNS``.

        Returns
        -------
        dict
            Keys: win_probability, predicted_margin, confidence,
            model_used, explanation.
        """
        if len(features_df) != 1:
            raise ValueError(
                f"features_df must have exactly 1 row, got {len(features_df)}"
            )

        missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = features_df[FEATURE_COLUMNS]

        # Calibrated ensemble probability
        cal_proba = self.calibrated_ensemble.predict_proba(X)[0, 1]

        # Predicted margin
        predicted_margin = float(self.xgb_reg.predict(X)[0])

        # Confidence: distance from 0.5, scaled to 0–1
        confidence = float(abs(cal_proba - 0.5) * 2.0)

        # SHAP explanation (top 5 drivers)
        explanation = self._get_shap_explanation(X, top_n=5)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "round_number": round_num,
            "win_probability": float(cal_proba),
            "predicted_margin": round(predicted_margin, 1),
            "confidence": round(confidence, 3),
            "model_used": "calibrated_ensemble (XGB + LGBM, Platt-scaled)",
            "explanation": explanation,
        }


def predict_match(
    home_team: str,
    away_team: str,
    venue: str,
    round_num: int,
    features_df: pd.DataFrame,
) -> dict[str, Any]:
    """Convenience function — creates a predictor and returns a prediction.

    For repeated predictions, prefer instantiating ``ModelPredictor`` once
    and calling ``predict_match`` on it to avoid reloading models each time.
    """
    predictor = ModelPredictor()
    return predictor.predict_match(
        home_team, away_team, venue, round_num, features_df
    )
