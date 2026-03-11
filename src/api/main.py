"""
AFL Match Predictor API.

FastAPI application serving match predictions, team info,
model metadata, and monitoring endpoints.

Run with: uvicorn src.api.main:app --reload
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests as http_requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.schemas import (
    AccuracyReport,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    MatchRequest,
    ModelInfo,
    PredictionResponse,
    SimulationRequest,
    TeamInfo,
)
from src.data.clean import standardise_team_name
from src.simulator.season import simulate_season

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DATA_DIR = PROJECT_ROOT / "data"

AFL_TEAMS = [
    "Adelaide",
    "Brisbane",
    "Carlton",
    "Collingwood",
    "Essendon",
    "Fremantle",
    "Geelong",
    "Gold Coast",
    "Greater Western Sydney",
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


# ── Rate limiter ─────────────────────────────────────────────────────

# Per-IP request tracking: {ip: [timestamp, timestamp, ...]}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)

# General: 60 requests per minute per IP
_RATE_LIMIT_GENERAL = 60
_RATE_LIMIT_WINDOW = 60.0

# Heavy endpoints: 5 requests per minute per IP
_RATE_LIMIT_HEAVY = 5
_HEAVY_PATHS = {"/simulate", "/explain"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter by client IP."""

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        path = request.url.path

        # Skip rate limiting for static files and health checks
        if path.startswith("/static") or path == "/health" or path == "/":
            response = await call_next(request)
            return self._add_security_headers(response)

        # Determine limit for this path
        is_heavy = path in _HEAVY_PATHS
        limit = _RATE_LIMIT_HEAVY if is_heavy else _RATE_LIMIT_GENERAL
        key = f"{client_ip}:{'heavy' if is_heavy else 'general'}"

        # Clean old entries and check
        _rate_limit_store[key] = [
            t for t in _rate_limit_store[key] if now - t < _RATE_LIMIT_WINDOW
        ]

        if len(_rate_limit_store[key]) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={
                    "Retry-After": str(int(_RATE_LIMIT_WINDOW)),
                    "X-RateLimit-Limit": str(limit),
                },
            )

        _rate_limit_store[key].append(now)

        response = await call_next(request)
        return self._add_security_headers(response)

    @staticmethod
    def _add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app = FastAPI(
    title="AFL Match Predictor API",
    description=(
        "Machine-learning powered AFL match prediction service. "
        "Provides win probabilities, margin estimates, and SHAP-based explanations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state populated at startup
_model_loaded: bool = False
_predictor = None
_tracker = None
_features_df: pd.DataFrame | None = None

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff", "elo_prob",
    "home_form_5", "away_form_5", "home_form_10", "away_form_10",
    "home_avg_margin_5", "away_avg_margin_5",
    "home_avg_score_5", "away_avg_score_5",
    "home_avg_conceded_5", "away_avg_conceded_5",
    "home_streak", "away_streak",
    "h2h_home_wins_5", "h2h_avg_margin_5",
    "home_venue_win_rate", "away_venue_win_rate",
    "round_number",
]


def _build_feature_row(
    home_team: str, away_team: str, venue: str, round_number: int,
) -> pd.DataFrame:
    """Build a feature row for prediction using the latest available data.

    Looks up the most recent feature values for each team from the features CSV.
    Falls back to sensible defaults if no data is available.
    """
    defaults = {col: 0.0 for col in FEATURE_COLUMNS}
    defaults.update({"home_elo": 1500.0, "away_elo": 1500.0, "elo_prob": 0.5, "round_number": round_number})

    if _features_df is None or _features_df.empty:
        return pd.DataFrame([defaults])

    # Find most recent game for each team to get their latest stats
    home_rows = _features_df[_features_df["home_team"] == home_team].sort_values("date", ascending=False)
    away_rows = _features_df[_features_df["away_team"] == away_team].sort_values("date", ascending=False)

    row = dict(defaults)
    row["round_number"] = float(round_number)

    if not home_rows.empty:
        latest = home_rows.iloc[0]
        for col in ["home_elo", "home_form_5", "home_form_10", "home_avg_margin_5",
                     "home_avg_score_5", "home_avg_conceded_5", "home_streak",
                     "home_venue_win_rate"]:
            if col in latest.index and pd.notna(latest[col]):
                row[col] = float(latest[col])

    if not away_rows.empty:
        latest = away_rows.iloc[0]
        for col in ["away_elo", "away_form_5", "away_form_10", "away_avg_margin_5",
                     "away_avg_score_5", "away_avg_conceded_5", "away_streak",
                     "away_venue_win_rate"]:
            if col in latest.index and pd.notna(latest[col]):
                row[col] = float(latest[col])

    # Compute derived features
    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    row["elo_prob"] = 1 / (1 + 10 ** ((-row["elo_diff"] - 40) / 400))

    # H2H features from most recent matchup
    h2h = _features_df[
        ((_features_df["home_team"] == home_team) & (_features_df["away_team"] == away_team))
        | ((_features_df["home_team"] == away_team) & (_features_df["away_team"] == home_team))
    ].sort_values("date", ascending=False)
    if not h2h.empty:
        row["h2h_home_wins_5"] = float(h2h.iloc[0].get("h2h_home_wins_5", 0))
        row["h2h_avg_margin_5"] = float(h2h.iloc[0].get("h2h_avg_margin_5", 0))

    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.on_event("startup")
async def load_models() -> None:
    """Attempt to load the trained predictor and monitoring tracker at startup."""
    global _model_loaded, _predictor, _tracker, _features_df  # noqa: PLW0603

    # Load features data for building prediction inputs
    features_path = DATA_DIR / "processed" / "features.csv"
    if features_path.exists():
        try:
            _features_df = pd.read_csv(features_path, parse_dates=["date"])
            logger.info("Features data loaded (%d rows).", len(_features_df))
        except Exception as exc:
            logger.warning("Could not load features data: %s", exc)

    # Try loading the predictor
    try:
        from src.models.predict import ModelPredictor

        _predictor = ModelPredictor()
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


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {
        "message": "AFL Match Predictor API. Visit /docs for API documentation.",
        "debug_static_dir": str(STATIC_DIR),
        "static_dir_exists": STATIC_DIR.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MatchRequest) -> PredictionResponse:
    """Return a win-probability prediction for the requested match."""
    _validate_team(request.home_team)
    _validate_team(request.away_team)

    if not _model_loaded or _predictor is None:
        return _mock_prediction(request)

    try:
        features_row = _build_feature_row(
            request.home_team, request.away_team, request.venue, request.round_number,
        )

        result = _predictor.predict_match(
            home_team=request.home_team,
            away_team=request.away_team,
            venue=request.venue,
            round_num=request.round_number,
            features_df=features_row,
        )

        home_prob = float(result.get("win_probability", 0.5))
        away_prob = round(1.0 - home_prob, 4)
        margin = float(result.get("predicted_margin", 0.0))

        top_features = [
            FeatureContribution(feature=f["feature"], contribution=f["shap_value"])
            for f in result.get("explanation", [])
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
        # Use ensemble metrics as the primary model metrics
        ensemble = data.get("ensemble_raw", data.get("xgb_classifier", {}))
        return ModelInfo(
            model_type="XGBoost + LightGBM Ensemble (Platt-calibrated)",
            accuracy=float(ensemble.get("accuracy", 0.0)),
            log_loss=float(ensemble.get("log_loss", 0.0)),
            last_trained=data.get("last_trained", "2025-03-10"),
            n_training_samples=1447,
            feature_count=len(FEATURE_COLUMNS),
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
        # Use XGBoost classifier importance (already sorted descending)
        xgb_importance = data.get("xgb_classifier", {})
        return [
            FeatureContribution(feature=feat, contribution=round(imp, 4))
            for feat, imp in xgb_importance.items()
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
        features_row = _build_feature_row(
            request.home_team, request.away_team, request.venue, request.round_number,
        )

        result = _predictor.predict_match(
            home_team=request.home_team,
            away_team=request.away_team,
            venue=request.venue,
            round_num=request.round_number,
            features_df=features_row,
        )

        shap_values = [
            FeatureContribution(feature=sv["feature"], contribution=sv["shap_value"])
            for sv in result.get("explanation", [])
        ]

        return ExplainResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            base_value=0.5,
            shap_values=shap_values,
            prediction=float(result.get("win_probability", 0.5)),
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


# ---------------------------------------------------------------------------
# Squiggle API cache & helpers
# ---------------------------------------------------------------------------

_squiggle_cache: dict[int, dict] = {}
_squiggle_cache_ts: dict[int, float] = {}
_CACHE_TTL = 300  # 5 minutes
_SQUIGGLE_RATE_LIMIT_INTERVAL = 1.0  # seconds between requests
_last_squiggle_request: float = 0.0


def _fetch_squiggle_games(year: int) -> list[dict]:
    """Fetch games from the Squiggle API with caching and rate limiting."""
    global _last_squiggle_request  # noqa: PLW0603

    now = time.time()

    # Return cached data if still fresh
    if year in _squiggle_cache and (now - _squiggle_cache_ts.get(year, 0)) < _CACHE_TTL:
        return _squiggle_cache[year]

    # Rate limit
    elapsed = now - _last_squiggle_request
    if elapsed < _SQUIGGLE_RATE_LIMIT_INTERVAL:
        time.sleep(_SQUIGGLE_RATE_LIMIT_INTERVAL - elapsed)

    try:
        resp = http_requests.get(
            "https://api.squiggle.com.au",
            params={"q": "games", "year": year},
            headers={"User-Agent": "AFL-Predict/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
        games = resp.json().get("games", [])
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data from Squiggle API: {exc}",
        ) from exc
    finally:
        _last_squiggle_request = time.time()

    _squiggle_cache[year] = games
    _squiggle_cache_ts[year] = time.time()
    return games


def _split_played_upcoming(games: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate Squiggle games into played results and upcoming matches."""
    played = []
    upcoming = []
    for g in games:
        # Squiggle marks complete games with a non-zero 'complete' field or a score
        is_complete = g.get("complete", 0) == 100 or (
            g.get("hscore") is not None
            and g.get("ascore") is not None
            and g.get("hscore", 0) > 0
        )
        home = standardise_team_name(g.get("hteam", ""))
        away = standardise_team_name(g.get("ateam", ""))
        if is_complete:
            played.append({
                "home_team": home,
                "away_team": away,
                "home_score": g.get("hscore", 0),
                "away_score": g.get("ascore", 0),
                "venue": g.get("venue", ""),
                "round_number": g.get("round", 0),
            })
        else:
            upcoming.append({
                "home_team": home,
                "away_team": away,
                "venue": g.get("venue", ""),
                "round_number": g.get("round", 0),
            })
    return played, upcoming


# ---------------------------------------------------------------------------
# ELO ladder
# ---------------------------------------------------------------------------


@app.get("/elo/ladder")
async def elo_ladder():
    """Return all 18 AFL teams ranked by current ELO rating."""
    if _features_df is None or _features_df.empty:
        raise HTTPException(status_code=404, detail="No features data available.")

    elo_data = {}
    for team in AFL_TEAMS:
        # Get latest ELO as home
        home = _features_df[_features_df["home_team"] == team].sort_values(
            "date", ascending=False
        )
        # Get latest ELO as away
        away = _features_df[_features_df["away_team"] == team].sort_values(
            "date", ascending=False
        )

        home_date = home.iloc[0]["date"] if not home.empty else pd.Timestamp.min
        away_date = away.iloc[0]["date"] if not away.empty else pd.Timestamp.min

        if home_date >= away_date and not home.empty:
            elo_data[team] = float(home.iloc[0]["home_elo"])
        elif not away.empty:
            elo_data[team] = float(away.iloc[0]["away_elo"])
        else:
            elo_data[team] = 1500.0

    # Sort by ELO descending
    sorted_teams = sorted(elo_data.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "rank": i + 1,
            "team": team,
            "elo": round(elo, 1),
            "diff_from_avg": round(elo - 1500, 1),
        }
        for i, (team, elo) in enumerate(sorted_teams)
    ]


# ---------------------------------------------------------------------------
# New endpoints — fixture, simulate, round predictions
# ---------------------------------------------------------------------------


@app.get("/fixture/{year}")
async def get_fixture(year: int):
    """Fetch the AFL fixture for a given year from the Squiggle API."""
    games = _fetch_squiggle_games(year)
    played, upcoming = _split_played_upcoming(games)
    return {
        "year": year,
        "total_games": len(games),
        "played": played,
        "upcoming": upcoming,
    }


@app.post("/simulate")
async def simulate(request: SimulationRequest):
    """Run a Monte Carlo season simulation and return ladder probabilities."""
    games = _fetch_squiggle_games(2026)
    played, upcoming = _split_played_upcoming(games)

    if not upcoming:
        raise HTTPException(
            status_code=400,
            detail="No upcoming matches found for 2026 — season may be complete.",
        )

    # Build a predict_fn compatible with simulate_season
    def predict_fn(home_team: str, away_team: str, venue: str, round_number: int) -> dict:
        canonical_home = standardise_team_name(home_team)
        canonical_away = standardise_team_name(away_team)

        if _model_loaded and _predictor is not None:
            features_row = _build_feature_row(canonical_home, canonical_away, venue, round_number)
            result = _predictor.predict_match(
                home_team=canonical_home,
                away_team=canonical_away,
                venue=venue,
                round_num=round_number,
                features_df=features_row,
            )
            return {
                "home_prob": float(result.get("win_probability", 0.5)),
                "home_elo": features_row["home_elo"].iloc[0] if "home_elo" in features_row.columns else 1500.0,
                "away_elo": features_row["away_elo"].iloc[0] if "away_elo" in features_row.columns else 1500.0,
            }
        else:
            # Fallback: simple ELO-based probability
            return {"home_prob": 0.5, "home_elo": 1500.0, "away_elo": 1500.0}

    try:
        sim_result = simulate_season(
            played_results=played,
            upcoming_matches=upcoming,
            predict_fn=predict_fn,
            n_simulations=request.n_simulations,
            stochastic=True,
        )

        # Convert DataFrames to serialisable dicts
        ladder = (
            sim_result["deterministic_ladder"].to_dict(orient="records")
            if hasattr(sim_result.get("deterministic_ladder", None), "to_dict")
            else sim_result.get("deterministic_ladder", [])
        )
        position_probs = (
            sim_result["position_probabilities"].to_dict(orient="records")
            if hasattr(sim_result.get("position_probabilities", None), "to_dict")
            else sim_result.get("position_probabilities", [])
        )

        return {
            "n_simulations": request.n_simulations,
            "played_games": len(played),
            "upcoming_games": len(upcoming),
            "deterministic_ladder": ladder,
            "position_probabilities": position_probs,
            "finals_probability": sim_result.get("finals_probability", {}),
            "premiership_probability": sim_result.get("premiership_probability", {}),
        }

    except Exception as exc:
        logger.error("Season simulation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Simulation error: {exc}"
        ) from exc


@app.get("/round/{year}/{round_number}/predictions")
async def round_predictions(year: int, round_number: int):
    """Return predictions for every match in a specific round."""
    games = _fetch_squiggle_games(year)

    round_games = [
        g for g in games
        if g.get("round") == round_number
    ]

    if not round_games:
        raise HTTPException(
            status_code=404,
            detail=f"No matches found for {year} round {round_number}.",
        )

    predictions = []
    for g in round_games:
        home = standardise_team_name(g.get("hteam", ""))
        away = standardise_team_name(g.get("ateam", ""))
        venue = g.get("venue", "")

        # Validate teams — skip if unknown
        try:
            _validate_team(home)
            _validate_team(away)
        except HTTPException:
            predictions.append({
                "home_team": home,
                "away_team": away,
                "venue": venue,
                "error": f"Unknown team name: {home} or {away}",
            })
            continue

        if _model_loaded and _predictor is not None:
            try:
                features_row = _build_feature_row(home, away, venue, round_number)
                result = _predictor.predict_match(
                    home_team=home,
                    away_team=away,
                    venue=venue,
                    round_num=round_number,
                    features_df=features_row,
                )
                home_prob = float(result.get("win_probability", 0.5))
                away_prob = round(1.0 - home_prob, 4)
                margin = float(result.get("predicted_margin", 0.0))
                top_features = [
                    {"feature": f["feature"], "contribution": f["shap_value"]}
                    for f in result.get("explanation", [])
                ]

                predictions.append({
                    "home_team": home,
                    "away_team": away,
                    "venue": venue,
                    "home_prob": round(home_prob, 4),
                    "away_prob": away_prob,
                    "predicted_margin": round(margin, 1),
                    "confidence": _confidence_label(home_prob),
                    "top_features": top_features,
                })
            except Exception as exc:
                predictions.append({
                    "home_team": home,
                    "away_team": away,
                    "venue": venue,
                    "error": str(exc),
                })
        else:
            # Mock prediction
            random.seed(hash((home, away, venue)))
            home_prob = round(random.uniform(0.30, 0.70), 4)
            predictions.append({
                "home_team": home,
                "away_team": away,
                "venue": venue,
                "home_prob": home_prob,
                "away_prob": round(1.0 - home_prob, 4),
                "predicted_margin": round((home_prob - 0.5) * 80, 1),
                "confidence": _confidence_label(home_prob),
                "top_features": [
                    {"feature": "home_elo", "contribution": 0.15},
                    {"feature": "away_elo", "contribution": -0.10},
                ],
                "note": "Mock prediction — model not yet trained.",
            })

    return {
        "year": year,
        "round_number": round_number,
        "matches": len(predictions),
        "predictions": predictions,
    }


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
