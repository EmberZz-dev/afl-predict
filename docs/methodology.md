# Methodology

## 1. Problem Statement

Predict AFL match outcomes (win/loss and margin) using historical match data and engineered features. Serve predictions via a REST API with model explainability.

## 2. Data

**Source:** Squiggle API (https://api.squiggle.com.au) — free, public AFL data.

**Coverage:** 2015–2025 seasons (~2,000+ matches)

**Raw features per match:**
- Date, round, season
- Home team, away team
- Home score, away score
- Venue

## 3. Feature Engineering

All features are computed using **only data available before the match** (no leakage).

### 3.1 ELO Rating System

Custom implementation based on chess ELO adapted for AFL:

```
Expected_home = 1 / (1 + 10^((away_elo - home_elo - HGA) / 400))

New_elo = Old_elo + K * (Actual - Expected)
```

Parameters:
- **K-factor:** 30 (balances responsiveness vs stability)
- **Home Ground Advantage (HGA):** +40 ELO points
- **Season regression:** 20% toward 1500 between seasons (prevents rating drift)

**Why ELO?** It's a proven, interpretable strength metric. Quant firms use similar rating systems for relative value.

### 3.2 Form Features

Rolling statistics over recent games capture team momentum:

| Feature | Window | Description |
|---------|--------|-------------|
| Win rate | 5, 10 games | Fraction of recent wins |
| Avg margin | 5 games | Average winning/losing margin |
| Avg score | 5 games | Offensive strength |
| Avg conceded | 5 games | Defensive strength |
| Streak | Current | Consecutive wins (+) or losses (-) |

### 3.3 Head-to-Head

Historical matchup data between the two teams:
- Win rate in last 5 meetings
- Average margin in last 5 meetings

### 3.4 Venue Effects

Ground-specific performance:
- Team's win rate at this venue (3-year window)
- Whether it's the team's designated home ground

### 3.5 Rest & Scheduling

- Days since last game (short turnarounds disadvantage teams)
- Round number (early vs late season dynamics)

## 4. Model Selection

### Why XGBoost + LightGBM?

- Gradient boosting dominates tabular data tasks
- Both handle missing values naturally
- Feature importance is built-in
- Fast inference for real-time API

### Training Protocol

1. **Temporal split:** Train on 2015–2023, test on 2024+
   - No random split — respects time ordering (critical for financial ML)
2. **Hyperparameter tuning:** GridSearchCV with 5-fold time-series CV
3. **Calibration:** Platt scaling via CalibratedClassifierCV
   - Raw probabilities from tree models are often poorly calibrated
   - Calibration ensures "70% predictions" actually win ~70% of the time

### Ensemble

Simple average of XGBoost and LightGBM probabilities. Ensembles reduce variance and typically outperform individual models.

## 5. Evaluation Metrics

| Metric | What it measures |
|--------|------------------|
| **Accuracy** | Overall correctness |
| **Log Loss** | Probability quality (penalizes confident wrong predictions) |
| **Brier Score** | Calibration quality |
| **Calibration Curve** | Visual check of probability reliability |
| **MAE (margin)** | Margin prediction error |

### Why these metrics matter for quant:
- **Log loss** is analogous to information ratio in finance
- **Calibration** is essential for any probabilistic system (options pricing, risk)
- **Temporal validation** mirrors walk-forward analysis in backtesting

## 6. Model Explainability

Every prediction includes SHAP (SHapley Additive exPlanations) values:
- Which features drove this specific prediction
- Direction and magnitude of each feature's contribution
- Consistent with game-theoretic fairness properties

## 7. Monitoring & Drift Detection

Post-deployment tracking:
- Rolling accuracy over last 20 and 50 predictions
- Alert if accuracy drops below 55% (potential model drift)
- Log all predictions for future retraining

## 8. Limitations & Future Work

**Current limitations:**
- No player-level data (injuries, suspensions, team changes)
- No weather data
- No betting market data (could be a strong feature)
- Simple ensemble — could explore stacking or neural approaches

**Future improvements:**
- Add player-level features from AFL Stats
- Incorporate live betting odds as a feature
- Build a Streamlit dashboard for visualization
- Retrain automatically when new season data arrives
- A/B test model versions
