# 8-Week Roadmap (5 hours/week)

## Week 1: Data Collection & EDA (5 hrs)
- [ ] Set up GitHub repo, virtual environment
- [ ] Run `python -m src.data.collect` to fetch AFL data
- [ ] Run `python -m src.data.clean` to clean data
- [ ] Create `notebooks/01_eda.ipynb` — explore the data:
  - How many matches per season?
  - Home win rate (should be ~57-58%)
  - Score distributions
  - Which venues have strongest home advantage?
- [ ] First commit + push to GitHub

**Deliverable:** Clean dataset + EDA notebook

## Week 2: Feature Engineering (5 hrs)
- [ ] Run `python -m src.features.build` to generate features
- [ ] Create `notebooks/02_features.ipynb` — analyze features:
  - ELO rating distributions and evolution over time
  - Correlation matrix of all features
  - Which features correlate most with outcomes?
  - Check for data leakage (correlation > 0.95 is suspicious)
- [ ] Verify no NaN leakage in feature matrix
- [ ] Commit + push

**Deliverable:** Feature matrix + feature analysis notebook

## Week 3: Model Training v1 (5 hrs)
- [ ] Run `python -m src.models.train`
- [ ] Create `notebooks/03_model_v1.ipynb`:
  - Baseline: always predict home team wins (~57% accuracy)
  - ELO-only model accuracy
  - Full model accuracy
  - Feature importance plot
  - Confusion matrix
- [ ] Document results in notebook
- [ ] Commit + push

**Deliverable:** Trained models + evaluation notebook

## Week 4: Model Improvement & Calibration (5 hrs)
- [ ] Analyze misclassified games — what went wrong?
- [ ] Try additional features if needed
- [ ] Implement calibration and verify with calibration curve
- [ ] Create `notebooks/04_calibration.ipynb`:
  - Before/after calibration comparison
  - Reliability diagram
  - Brier score improvement
- [ ] Commit + push

**Deliverable:** Calibrated model + analysis notebook

## Week 5: API Development (5 hrs)
- [ ] Start FastAPI server: `uvicorn src.api.main:app --reload`
- [ ] Test all endpoints with Swagger UI (localhost:8000/docs)
- [ ] Write integration tests in `tests/`
- [ ] Add error handling for invalid teams/venues
- [ ] Test with real upcoming AFL matches
- [ ] Commit + push

**Deliverable:** Working API with tests

## Week 6: SHAP Explanations & Monitoring (5 hrs)
- [ ] Verify SHAP explanations are meaningful
- [ ] Create `notebooks/05_explanations.ipynb`:
  - SHAP summary plot (all features)
  - SHAP force plots for specific predictions
  - SHAP dependence plots (ELO diff vs outcome)
- [ ] Set up monitoring tracker
- [ ] Run predictions on recent matches, check accuracy
- [ ] Commit + push

**Deliverable:** Explainability notebook + monitoring

## Week 7: Documentation & Testing (5 hrs)
- [ ] Write comprehensive README (already started)
- [ ] Clean up all notebooks (add markdown explanations)
- [ ] Ensure methodology.md is complete
- [ ] Add docstrings to all public functions
- [ ] Write unit tests for:
  - ELO calculations
  - Feature engineering (no leakage test)
  - API endpoints
- [ ] Achieve 80%+ test coverage on core modules
- [ ] Commit + push

**Deliverable:** Polished documentation + tests

## Week 8: Polish & Deploy (5 hrs)
- [ ] Create a simple HTML landing page or Streamlit dashboard (optional)
- [ ] Dockerize the API (Dockerfile)
- [ ] Record a short demo (screen recording or GIF in README)
- [ ] Write a "Results" section in README with final metrics
- [ ] Final review — would YOU be impressed seeing this on GitHub?
- [ ] Clean git history, tag v1.0.0
- [ ] Share on LinkedIn, add to resume

**Deliverable:** Portfolio-ready project

---

## What Recruiters Will Look For

1. **Clean code** — type hints, docstrings, consistent style
2. **Proper ML methodology** — temporal splits, calibration, no leakage
3. **End-to-end system** — not just a notebook, but a deployable API
4. **Testing** — shows production engineering mindset
5. **Documentation** — methodology doc shows you can communicate
6. **Git history** — regular commits show consistent work ethic

## Stretch Goals (if time permits)

- [ ] Streamlit dashboard with live predictions
- [ ] Docker + CI/CD pipeline
- [ ] Betting simulation: if you bet on every prediction, what's your ROI?
- [ ] Add player-level features
- [ ] Compare against bookmaker accuracy
- [ ] Deploy to Railway/Render (free tier)
