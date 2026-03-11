"""Pydantic models for the AFL Match Predictor API."""

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    """Input schema for a match prediction request."""

    home_team: str = Field(..., description="Name of the home team")
    away_team: str = Field(..., description="Name of the away team")
    venue: str = Field(..., description="Match venue")
    round_number: int = Field(..., ge=1, le=27, description="Round number in the season")


class FeatureContribution(BaseModel):
    """A single feature's contribution to the prediction."""

    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    """Output schema for a match prediction."""

    home_team: str
    away_team: str
    home_win_probability: float = Field(..., ge=0.0, le=1.0)
    away_win_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_margin: float
    confidence: str = Field(..., pattern="^(high|medium|low)$")
    top_features: list[FeatureContribution]
    note: str | None = None


class TeamInfo(BaseModel):
    """Information about a single AFL team."""

    name: str
    elo_rating: float
    form_last_5: float = Field(..., ge=0.0, le=1.0, description="Win rate over last 5 games")
    rank: int = Field(..., ge=1, le=18)


class ModelInfo(BaseModel):
    """Metadata and performance metrics for the trained model."""

    model_type: str
    accuracy: float
    log_loss: float
    last_trained: str
    n_training_samples: int
    feature_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


class ExplainResponse(BaseModel):
    """Detailed SHAP explanation for a prediction."""

    home_team: str
    away_team: str
    base_value: float
    shap_values: list[FeatureContribution]
    prediction: float


class AccuracyReport(BaseModel):
    """Rolling accuracy report from the monitoring tracker."""

    rolling_20: float | None = None
    rolling_50: float | None = None
    all_time: float | None = None
    total_predictions: int
    drift_detected: bool


class SimulationRequest(BaseModel):
    """Input schema for a season simulation request."""

    n_simulations: int = Field(
        default=1000, ge=1, le=100000, description="Number of Monte Carlo simulations"
    )
