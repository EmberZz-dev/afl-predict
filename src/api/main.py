"""
AFL Match Predictor API.

FastAPI application serving match predictions, team info,
model metadata, and monitoring endpoints.

Run with: uvicorn src.api.main:app --reload
"""

import json
import logging
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AccuracyReport,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    MatchRequest,
    ModelInfo,
    PredictionResponse,
    TeamInfo,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DATA_DIR = PROJECT_ROOT / "data"

AFL_TEAMS = [
    "Adelaide",
    "Brisbane Lions",
    "Carlton",
    "Collingwood",
    "Essendon",
    "Fremantle",
    "Geelong",
    "Gold Coast",
    "GWS Giants",
    "Hawthorn",
    "Melbourne",
    "North Melbourne",
    "Port Adelaide",
    "Richmond",
    "St Kilda",
    "Sydney",
    "West Coast",
    "Western Bulldogs",
]

# Default ELO ratings and form (used when no trained model is available)
_DEFAULT_ELO: dict[str, float] = {team: 1500.0 for team in AFL_TEAMS}
_DEFAULT_FORM: dict[str, float] = {team: 0.5 for team in AFL_TEAMS}

app = FastAPI(
    title="AFL Match Predictor API",
    description=(
        "Machine-learning powered AFL match prediction service. "
        "Provides win probabilities, margin estimates, and SHAP-based explanations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state populated at startup
_model_loaded: bool = False
_predictor = None
_tracker = None


@app.on_event("startup")
async def load_models() -> None:
    """Attempt to load the trained predictor and monitoring tracker at startup."""
    global _model_loaded, _predictor, _tracker  # noqa: PLW0603

    # Try loading the predictor
    try:
        from src.models.predict import MatchPredictor

        predictor = MatchPredictor()
        predictor.load()
        _predictor = predictor
        _model_loaded = True
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning("Could not load model — mock predictions will be used: %s", exc)
        _model_loaded = False

    # Initialise monitoring tracker
    try:
        from src.monitoring.tracker import PredictionTracker

        _tracker = PredictionTracker()
        logger.info("Prediction tracker initialised.")
    except Exception as exc:
        logger.warning("Could not initialise prediction tracker: %s", exc)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _confidence_label(prob: float) -> str:
    """Return a confidence label based on distance from 0.5."""
    distance = abs(prob - 0.5)
    if distance >= 0.25:
        return "high"
    if distance >= 0.10:
        return "medium"
    return "low"


def _mock_prediction(request: MatchRequest) -> PredictionResponse:
    """Generate a plausible mock prediction when no model is available."""
    random.seed(hash((request.home_team, request.away_team, request.venue)))
    home_prob = round(random.uniform(0.30, 0.70), 4)
    away_prob = round(1.0 - home_prob, 4)
    margin = round((home_prob - 0.5) * 80, 1)

    return PredictionResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        home_win_probability=home_prob,
        away_win_probability=away_prob,
        predicted_margin=margin,
        confidence=_confidence_label(home_prob),
        top_features=[
            FeatureContribution(feature="home_elo", contribution=0.15),
            FeatureContribution(feature="away_elo", contribution=-0.10),
            FeatureContribution(feature="venue_advantage", contribution=0.08),
            FeatureContribution(feature="recent_form", contribution=0.06),
            FeatureContribution(feature="head_to_head", contribution=0.04),
        ],
        note="Mock prediction — model not yet trained.",
    )


def _validate_team(team_name: str) -> None:
    """Raise 400 if the team name is not a valid AFL team."""
    if team_name not in AFL_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown team '{team_name}'. Valid teams: {', '.join(AFL_TEAMS)}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MatchRequest) -> PredictionResponse:
    """Return a win-probability prediction for the requested match."""
    _validate_team(request.home_team)
    _validate_team(request.away_team)

    if not _model_loaded or _predictor is None:
        return _mock_prediction(request)

    try:
        result = _predictor.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            venue=request.venue,
            round_number=request.round_number,
        )

        home_prob = float(result.get("home_win_probability", 0.5))
        away_prob = round(1.0 - home_prob, 4)
        margin = float(result.get("predicted_margin", 0.0))

        top_features = [
            FeatureContribution(feature=f["feature"], contribution=f["contribution"])
            for f in result.get("top_features", [])
        ]

        response = PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            home_win_probability=round(home_prob, 4),
            away_win_probability=away_prob,
            predicted_margin=round(margin, 1),
            confidence=_confidence_label(home_prob),
            top_features=top_features,
        )

        # Log to tracker
        if _tracker is not None:
            _tracker.log_prediction(
                home_team=request.home_team,
                away_team=request.away_team,
                venue=request.venue,
                round_number=request.round_number,
                predicted_winner=request.home_team if home_prob > 0.5 else request.away_team,
                home_win_prob=home_prob,
            )

        return response

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.get("/teams", response_model=list[TeamInfo])
async def list_teams() -> list[TeamInfo]:
    """Return all 18 AFL teams with current ELO ratings and form."""
    elo_ratings = dict(_DEFAULT_ELO)
    form_values = dict(_DEFAULT_FORM)

    # Try to load real ELO/form data if available
    elo_path = MODELS_DIR / "elo_ratings.json"
    form_path = MODELS_DIR / "team_form.json"

    if elo_path.exists():
        try:
            elo_ratings.update(json.loads(elo_path.read_text()))
        except Exception:
            pass

    if form_path.exists():
        try:
            form_values.update(json.loads(form_path.read_text()))
        except Exception:
            pass

    # Sort by ELO descending to determine rank
    sorted_teams = sorted(AFL_TEAMS, key=lambda t: elo_ratings.get(t, 1500.0), reverse=True)

    return [
        TeamInfo(
            name=team,
            elo_rating=round(elo_ratings.get(team, 1500.0), 1),
            form_last_5=round(form_values.get(team, 0.5), 3),
            rank=rank,
        )
        for rank, team in enumerate(sorted_teams, start=1)
    ]


@app.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Return model metadata and test-set performance metrics."""
    metrics_path = MODELS_DIR / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No model metrics found. Train a model first.",
        )

    try:
        data = json.loads(metrics_path.read_text())
        return ModelInfo(
            model_type=data.get("model_type", "unknown"),
            accuracy=float(data.get("accuracy", 0.0)),
            log_loss=float(data.get("log_loss", 0.0)),
            last_trained=data.get("last_trained", "unknown"),
            n_training_samples=int(data.get("n_training_samples", 0)),
            feature_count=int(data.get("feature_count", 0)),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {exc}") from exc


@app.get("/model/features", response_model=list[FeatureContribution])
async def model_features() -> list[FeatureContribution]:
    """Return feature importance rankings from the trained model."""
    fi_path = MODELS_DIR / "feature_importance.json"

    if not fi_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No feature importance data found. Train a model first.",
        )

    try:
        data = json.loads(fi_path.read_text())
        return [
            FeatureContribution(feature=item["feature"], contribution=item["importance"])
            for item in data
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error reading feature importance: {exc}"
        ) from exc


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: MatchRequest) -> ExplainResponse:
    """Return a detailed SHAP-based explanation for a match prediction."""
    _validate_team(request.home_team)
    _validate_team(request.away_team)

    if not _model_loaded or _predictor is None:
        # Return mock SHAP explanation
        random.seed(hash((request.home_team, request.away_team, request.venue)))
        base_value = 0.5
        shap_values = [
            FeatureContribution(feature="home_elo", contribution=round(random.uniform(-0.2, 0.2), 4)),
            FeatureContribution(feature="away_elo", contribution=round(random.uniform(-0.2, 0.2), 4)),
            FeatureContribution(feature="venue_advantage", contribution=round(random.uniform(0.0, 0.1), 4)),
            FeatureContribution(feature="recent_form_home", contribution=round(random.uniform(-0.15, 0.15), 4)),
            FeatureContribution(feature="recent_form_away", contribution=round(random.uniform(-0.15, 0.15), 4)),
            FeatureContribution(feature="head_to_head", contribution=round(random.uniform(-0.1, 0.1), 4)),
            FeatureContribution(feature="rest_days_diff", contribution=round(random.uniform(-0.05, 0.05), 4)),
        ]
        prediction = base_value + sum(sv.contribution for sv in shap_values)

        return ExplainResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            base_value=base_value,
            shap_values=shap_values,
            prediction=round(prediction, 4),
        )

    try:
        explanation = _predictor.explain(
            home_team=request.home_team,
            away_team=request.away_team,
            venue=request.venue,
            round_number=request.round_number,
        )

        shap_values = [
            FeatureContribution(feature=sv["feature"], contribution=sv["shap_value"])
            for sv in explanation.get("shap_values", [])
        ]

        return ExplainResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            base_value=float(explanation.get("base_value", 0.5)),
            shap_values=shap_values,
            prediction=float(explanation.get("prediction", 0.5)),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Explanation error: {exc}"
        ) from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_loaded,
        version="1.0.0",
    )


@app.get("/monitor/accuracy", response_model=AccuracyReport)
async def monitor_accuracy() -> AccuracyReport:
    """Return rolling accuracy metrics from the prediction tracker."""
    if _tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Monitoring tracker not initialised.",
        )

    try:
        report = _tracker.get_report()
        return AccuracyReport(
            rolling_20=report.get("rolling_20"),
            rolling_50=report.get("rolling_50"),
            all_time=report.get("all_time"),
            total_predictions=report.get("total_predictions", 0),
            drift_detected=report.get("drift_detected", False),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error generating accuracy report: {exc}"
        ) from exc
