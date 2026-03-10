# AFL Match Outcome Predictor

A real-time machine learning prediction system for Australian Rules Football match outcomes. End-to-end ML pipeline from data collection to production API with monitoring.

## What This Does

Predicts AFL match outcomes (win/loss + margin) using historical match data, player statistics, and engineered features. Serves predictions via a FastAPI REST API with real-time monitoring.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Layer  │────▶│  Features    │────▶│   Models     │────▶│   API        │
│              │     │              │     │              │     │              │
│ • AFL API    │     │ • ELO Rating │     │ • XGBoost    │     │ • FastAPI    │
│ • Web Scrape │     │ • Form Guide │     │ • LightGBM   │     │ • Predict    │
│ • CSV Ingest │     │ • H2H Stats  │     │ • Ensemble   │     │ • Explain    │
│              │     │ • Venue Adj  │     │ • Calibration│     │ • Monitor    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                                               ┌──────────────┐
                                                               │  Dashboard   │
                                                               │ • Accuracy   │
                                                               │ • Calibration│
                                                               │ • Feature    │
                                                               │   Importance │
                                                               └──────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Collect data
python -m src.data.collect

# Build features
python -m src.features.build

# Train models
python -m src.models.train

# Start API
uvicorn src.api.main:app --reload

# View docs
open http://localhost:8000/docs
```

## Project Structure

```
afl-predict/
├── src/
│   ├── data/           # Data collection & cleaning
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # Model training, evaluation, selection
│   ├── api/            # FastAPI prediction service
│   └── monitoring/     # Model performance tracking
├── tests/              # Unit & integration tests
├── notebooks/          # EDA & experimentation
├── data/
│   ├── raw/            # Raw collected data
│   └── processed/      # Cleaned & feature-engineered data
├── models/saved/       # Serialized trained models
└── docs/               # Documentation & methodology
```

## Key Features

- **Custom ELO Rating System** — Dynamic team strength ratings updated after each match
- **Form-Based Features** — Rolling averages, streaks, momentum indicators
- **Venue Intelligence** — Home/away adjustments, ground-specific performance
- **Ensemble Prediction** — Stacked XGBoost + LightGBM with calibrated probabilities
- **SHAP Explanations** — Every prediction includes feature-level explanations
- **Model Monitoring** — Track accuracy, calibration, and drift over time

## Tech Stack

- **Data**: pandas, requests, BeautifulSoup
- **ML**: scikit-learn, XGBoost, LightGBM, SHAP
- **API**: FastAPI, Pydantic, uvicorn
- **Monitoring**: matplotlib, custom metrics tracker
- **Testing**: pytest
- **Docs**: Jupyter notebooks, markdown

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict match outcome |
| GET | `/teams` | List all teams with current ELO |
| GET | `/model/info` | Model metadata & performance |
| GET | `/model/features` | Feature importance rankings |
| POST | `/explain` | SHAP explanation for a prediction |
| GET | `/health` | API health check |
| GET | `/monitor/accuracy` | Rolling accuracy metrics |

## Methodology

See [docs/methodology.md](docs/methodology.md) for detailed explanation of:
- Feature engineering decisions
- Model selection process
- Evaluation framework
- Calibration approach
