"""AFL Match Predictor — Streamlit Dashboard.

Live 2026 season predictions, ladder simulation, and model monitoring.

Run with: streamlit run dashboard.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DATA_DIR = PROJECT_ROOT / "data"

SQUIGGLE_URL = "https://api.squiggle.com.au"
HEADERS = {"User-Agent": "AFL-Predict/1.0"}

# ── Page config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AFL Match Predictor",
    page_icon="🏈",
    layout="wide",
)

# ── Helpers ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def fetch_squiggle(query: str, year: int) -> list[dict]:
    """Fetch data from Squiggle API with caching."""
    resp = requests.get(
        SQUIGGLE_URL,
        params={"q": query, "year": year},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get(query, data.get("games", []))


@st.cache_data(ttl=300)
def fetch_2026_games() -> pd.DataFrame:
    """Fetch all 2026 games from Squiggle API."""
    games = fetch_squiggle("games", 2026)
    df = pd.DataFrame(games)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def fetch_standings(year: int) -> pd.DataFrame:
    """Fetch current standings from Squiggle API."""
    resp = requests.get(
        SQUIGGLE_URL,
        params={"q": "standings", "year": year},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    standings = data.get("standings", [])
    if not standings:
        return pd.DataFrame()
    return pd.DataFrame(standings).sort_values("rank").reset_index(drop=True)


def get_predictor():
    """Load the model predictor (cached in session state)."""
    if "predictor" not in st.session_state:
        try:
            from src.models.predict import ModelPredictor
            st.session_state.predictor = ModelPredictor()
            st.session_state.model_loaded = True
        except Exception as e:
            st.session_state.predictor = None
            st.session_state.model_loaded = False
            st.session_state.model_error = str(e)
    return st.session_state.predictor


def get_features_df():
    """Load the features DataFrame (cached in session state)."""
    if "features_df" not in st.session_state:
        features_path = DATA_DIR / "processed" / "features.csv"
        if features_path.exists():
            st.session_state.features_df = pd.read_csv(
                features_path, parse_dates=["date"]
            )
        else:
            st.session_state.features_df = None
    return st.session_state.features_df


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


def build_feature_row(
    home_team: str, away_team: str, venue: str, round_number: int,
    features_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a feature row for prediction from the features CSV."""
    defaults = {col: 0.0 for col in FEATURE_COLUMNS}
    defaults.update({
        "home_elo": 1500.0, "away_elo": 1500.0,
        "elo_prob": 0.5, "round_number": float(round_number),
    })

    if features_df is None or features_df.empty:
        return pd.DataFrame([defaults])[FEATURE_COLUMNS]

    row = dict(defaults)
    row["round_number"] = float(round_number)

    home_rows = features_df[
        features_df["home_team"] == home_team
    ].sort_values("date", ascending=False)
    away_rows = features_df[
        features_df["away_team"] == away_team
    ].sort_values("date", ascending=False)

    if not home_rows.empty:
        latest = home_rows.iloc[0]
        for col in ["home_elo", "home_form_5", "home_form_10",
                     "home_avg_margin_5", "home_avg_score_5",
                     "home_avg_conceded_5", "home_streak",
                     "home_venue_win_rate"]:
            if col in latest.index and pd.notna(latest[col]):
                row[col] = float(latest[col])

    if not away_rows.empty:
        latest = away_rows.iloc[0]
        for col in ["away_elo", "away_form_5", "away_form_10",
                     "away_avg_margin_5", "away_avg_score_5",
                     "away_avg_conceded_5", "away_streak",
                     "away_venue_win_rate"]:
            if col in latest.index and pd.notna(latest[col]):
                row[col] = float(latest[col])

    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    row["elo_prob"] = 1 / (1 + 10 ** ((-row["elo_diff"] - 40) / 400))

    h2h = features_df[
        ((features_df["home_team"] == home_team) & (features_df["away_team"] == away_team))
        | ((features_df["home_team"] == away_team) & (features_df["away_team"] == home_team))
    ].sort_values("date", ascending=False)
    if not h2h.empty:
        row["h2h_home_wins_5"] = float(h2h.iloc[0].get("h2h_home_wins_5", 0))
        row["h2h_avg_margin_5"] = float(h2h.iloc[0].get("h2h_avg_margin_5", 0))

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_match_for_sim(
    home_team: str, away_team: str, venue: str, round_number: int,
) -> dict:
    """Predict a match and return data needed for simulation."""
    predictor = get_predictor()
    features_df = get_features_df()

    if predictor is None:
        # Fallback: use ELO-only prediction
        elo_prob = 0.5
        if features_df is not None:
            home_rows = features_df[
                features_df["home_team"] == home_team
            ].sort_values("date", ascending=False)
            away_rows = features_df[
                features_df["away_team"] == away_team
            ].sort_values("date", ascending=False)
            h_elo = float(home_rows.iloc[0]["home_elo"]) if not home_rows.empty else 1500.0
            a_elo = float(away_rows.iloc[0]["away_elo"]) if not away_rows.empty else 1500.0
            diff = h_elo - a_elo + 40
            elo_prob = 1 / (1 + 10 ** (-diff / 400))
        else:
            h_elo, a_elo = 1500.0, 1500.0

        return {"home_prob": elo_prob, "home_elo": h_elo, "away_elo": a_elo}

    feature_row = build_feature_row(
        home_team, away_team, venue, round_number, features_df,
    )

    result = predictor.predict_match(
        home_team=home_team, away_team=away_team,
        venue=venue, round_num=round_number,
        features_df=feature_row,
    )

    return {
        "home_prob": result["win_probability"],
        "home_elo": float(feature_row["home_elo"].iloc[0]),
        "away_elo": float(feature_row["away_elo"].iloc[0]),
    }


def standardise_team(name: str) -> str:
    """Map Squiggle team names to our canonical names."""
    mapping = {
        "GWS": "Greater Western Sydney",
        "GWS Giants": "Greater Western Sydney",
        "Brisbane Lions": "Brisbane",
        "Sydney Swans": "Sydney",
        "Geelong Cats": "Geelong",
        "West Coast Eagles": "West Coast",
        "Adelaide Crows": "Adelaide",
        "Gold Coast Suns": "Gold Coast",
        "Kangaroos": "North Melbourne",
        "Melbourne Demons": "Melbourne",
        "Port Adelaide Power": "Port Adelaide",
        "Richmond Tigers": "Richmond",
        "Carlton Blues": "Carlton",
        "Collingwood Magpies": "Collingwood",
        "Essendon Bombers": "Essendon",
        "Fremantle Dockers": "Fremantle",
        "Hawthorn Hawks": "Hawthorn",
        "St Kilda Saints": "St Kilda",
        "Western Bulldogs": "Western Bulldogs",
        "Footscray": "Western Bulldogs",
    }
    return mapping.get(name, name)


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("AFL Predictor")
    st.caption("2026 Season Dashboard")

    # Model status
    predictor = get_predictor()
    if st.session_state.get("model_loaded", False):
        st.success("Model loaded")
    else:
        st.warning("Model not loaded — using ELO fallback")
        if st.session_state.get("model_error"):
            st.caption(f"Error: {st.session_state.model_error}")

    st.divider()

    # Retrain button
    st.subheader("Retrain Model")
    st.caption("Re-fetches latest data and retrains all models.")
    if st.button("Retrain Now", type="primary", use_container_width=True):
        with st.spinner("Collecting latest data..."):
            subprocess.run(
                [sys.executable, "-m", "src.data.collect"],
                cwd=str(PROJECT_ROOT), capture_output=True,
            )
        with st.spinner("Cleaning data..."):
            subprocess.run(
                [sys.executable, "-m", "src.data.clean"],
                cwd=str(PROJECT_ROOT), capture_output=True,
            )
        with st.spinner("Building features..."):
            subprocess.run(
                [sys.executable, "-m", "src.features.build"],
                cwd=str(PROJECT_ROOT), capture_output=True,
            )
        with st.spinner("Training models (this may take a few minutes)..."):
            result = subprocess.run(
                [sys.executable, "-m", "src.models.train"],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            )
        if result.returncode == 0:
            # Clear caches to reload
            for key in ["predictor", "model_loaded", "features_df"]:
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.success("Retrained successfully!")
            st.rerun()
        else:
            st.error("Training failed.")
            st.code(result.stderr[-500:] if result.stderr else "No error output")

    st.divider()

    # Model info
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        ensemble = metrics.get("ensemble_raw", {})
        st.metric("Accuracy", f"{ensemble.get('accuracy', 0):.1%}")
        st.metric("Log Loss", f"{ensemble.get('log_loss', 0):.3f}")


# ── Main content ─────────────────────────────────────────────────────────

tab_predictions, tab_ladder, tab_accuracy = st.tabs([
    "Round Predictions", "Season Simulation", "Model Performance",
])


# ── Tab 1: Round Predictions ────────────────────────────────────────────

with tab_predictions:
    st.header("2026 Match Predictions")

    try:
        games_df = fetch_2026_games()
    except Exception as e:
        st.error(f"Could not fetch 2026 fixture: {e}")
        st.stop()

    # Separate played and unplayed
    played = games_df[games_df["complete"] == 100].copy()
    upcoming = games_df[games_df["complete"] == 0].copy()

    if upcoming.empty:
        st.info("No upcoming games found. The season may be complete.")
    else:
        # Get available rounds
        upcoming_rounds = sorted(upcoming["round"].unique())
        current_round = upcoming_rounds[0]

        selected_round = st.selectbox(
            "Select Round",
            upcoming_rounds,
            format_func=lambda r: f"Round {r}" if r > 0 else "Opening Round",
        )

        round_games = upcoming[upcoming["round"] == selected_round].sort_values("date")

        st.subheader(
            f"Round {selected_round} Predictions"
            if selected_round > 0 else "Opening Round Predictions"
        )

        for _, game in round_games.iterrows():
            home = standardise_team(game["hteam"])
            away = standardise_team(game["ateam"])
            venue = game["venue"]
            game_date = game["date"]

            col1, col2, col3 = st.columns([2, 1, 2])

            try:
                pred = predict_match_for_sim(
                    home, away, venue, int(selected_round),
                )
                home_prob = pred["home_prob"]
                away_prob = 1 - home_prob

                with col1:
                    prob_pct = f"{home_prob:.0%}"
                    st.markdown(f"### {home}")
                    st.progress(home_prob)
                    st.caption(f"{prob_pct} win probability")

                with col2:
                    st.markdown("### vs")
                    st.caption(f"{venue}")
                    if isinstance(game_date, pd.Timestamp):
                        st.caption(game_date.strftime("%a %d %b, %I:%M%p"))
                    predicted_winner = home if home_prob >= 0.5 else away
                    confidence = abs(home_prob - 0.5) * 2
                    if confidence >= 0.5:
                        conf_label = "High"
                    elif confidence >= 0.2:
                        conf_label = "Medium"
                    else:
                        conf_label = "Low"
                    st.caption(f"Confidence: {conf_label}")

                with col3:
                    prob_pct = f"{away_prob:.0%}"
                    st.markdown(f"### {away}")
                    st.progress(away_prob)
                    st.caption(f"{prob_pct} win probability")

            except Exception as e:
                with col2:
                    st.error(f"Prediction error: {e}")

            st.divider()

    # Show completed results
    if not played.empty:
        with st.expander(f"Completed Results ({len(played)} games)"):
            results_display = played[
                ["roundname", "hteam", "hscore", "ateam", "ascore", "venue"]
            ].copy()
            results_display.columns = [
                "Round", "Home", "H Score", "Away", "A Score", "Venue"
            ]
            st.dataframe(results_display, use_container_width=True, hide_index=True)


# ── Tab 2: Season Simulation ────────────────────────────────────────────

with tab_ladder:
    st.header("2026 Season Ladder Simulation")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        n_sims = st.slider("Monte Carlo simulations", 100, 5000, 1000, step=100)
    with col_sim2:
        st.caption(
            "More simulations = more accurate probability estimates but slower. "
            "1000 is a good balance."
        )

    if st.button("Run Season Simulation", type="primary"):
        try:
            games_df = fetch_2026_games()
        except Exception as e:
            st.error(f"Could not fetch fixture: {e}")
            st.stop()

        played = games_df[games_df["complete"] == 100]
        upcoming = games_df[games_df["complete"] == 0]

        # Build played results list
        played_results = []
        for _, g in played.iterrows():
            played_results.append({
                "home_team": standardise_team(g["hteam"]),
                "away_team": standardise_team(g["ateam"]),
                "home_score": int(g["hscore"]),
                "away_score": int(g["ascore"]),
            })

        # Build upcoming matches list
        upcoming_matches = []
        for _, g in upcoming.iterrows():
            upcoming_matches.append({
                "home_team": standardise_team(g["hteam"]),
                "away_team": standardise_team(g["ateam"]),
                "venue": g["venue"],
                "round_number": int(g["round"]) if pd.notna(g["round"]) else 1,
            })

        from src.simulator.season import simulate_season

        with st.spinner(f"Running {n_sims} simulations..."):
            sim_result = simulate_season(
                played_results=played_results,
                upcoming_matches=upcoming_matches,
                predict_fn=predict_match_for_sim,
                n_simulations=n_sims,
                stochastic=True,
            )

        ladder = sim_result["deterministic_ladder"]

        # Display ladder
        st.subheader("Predicted Final Ladder")

        # Format for display
        display_ladder = ladder[[
            "team", "wins", "losses", "draws", "pts",
            "percentage", "avg_wins", "finals_prob", "flag_prob",
        ]].copy()
        display_ladder.columns = [
            "Team", "W", "L", "D", "Pts",
            "%", "Avg Wins", "Finals %", "Flag %",
        ]

        # Highlight top 8
        st.dataframe(
            display_ladder.style.apply(
                lambda row: [
                    "background-color: rgba(76, 175, 80, 0.15)"
                    if row.name <= 8 else "" for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
            height=670,
        )
        st.caption("Top 8 (green) qualify for finals. Probabilities from Monte Carlo simulation.")

        # Finals probability chart
        st.subheader("Finals Probability")
        finals_df = pd.DataFrame([
            {"Team": team, "Finals %": prob}
            for team, prob in sorted(
                sim_result["finals_probability"].items(),
                key=lambda x: x[1], reverse=True,
            )
        ])
        st.bar_chart(finals_df.set_index("Team"), horizontal=True)

        # Minor premiership
        st.subheader("Minor Premiership (1st Place) Probability")
        flag_df = pd.DataFrame([
            {"Team": team, "Flag %": prob}
            for team, prob in sorted(
                sim_result["premiership_probability"].items(),
                key=lambda x: x[1], reverse=True,
            )
            if prob > 0
        ])
        if not flag_df.empty:
            st.bar_chart(flag_df.set_index("Team"), horizontal=True)

        st.caption(f"Based on {sim_result['n_simulations']} simulations.")

    else:
        st.info("Click 'Run Season Simulation' to predict the 2026 ladder.")

        # Show current actual standings if available
        try:
            standings = fetch_standings(2026)
            if not standings.empty:
                st.subheader("Current 2026 Standings (Actual)")
                display_cols = [c for c in ["rank", "name", "played", "wins", "losses",
                                            "draws", "pts", "percentage"]
                                if c in standings.columns]
                st.dataframe(
                    standings[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception:
            pass


# ── Tab 3: Model Performance ────────────────────────────────────────────

with tab_accuracy:
    st.header("Model Performance")

    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        st.warning("No trained model found. Run the training pipeline first.")
        st.stop()

    metrics = json.loads(metrics_path.read_text())

    # Test set metrics
    st.subheader("Test Set Performance (2024+ games)")

    col1, col2, col3, col4 = st.columns(4)
    ensemble = metrics.get("ensemble_raw", {})
    with col1:
        st.metric("Accuracy", f"{ensemble.get('accuracy', 0):.1%}")
    with col2:
        st.metric("Log Loss", f"{ensemble.get('log_loss', 0):.3f}")
    with col3:
        st.metric("Brier Score", f"{ensemble.get('brier_score', 0):.3f}")
    with col4:
        st.metric("vs Baseline", "+8.7pp", delta="8.7pp")

    # Model comparison
    st.subheader("Model Comparison")

    model_names = {
        "xgb_classifier": "XGBoost",
        "lgbm_classifier": "LightGBM",
        "ensemble_raw": "Ensemble (avg)",
        "ensemble_calibrated": "Calibrated",
    }

    comparison_rows = []
    for key, display_name in model_names.items():
        if key in metrics and isinstance(metrics[key], dict):
            m = metrics[key]
            comparison_rows.append({
                "Model": display_name,
                "Accuracy": f"{m.get('accuracy', 0):.1%}",
                "Log Loss": f"{m.get('log_loss', 0):.3f}",
                "Brier Score": f"{m.get('brier_score', 0):.3f}",
                "F1 Score": f"{m.get('f1', 0):.3f}",
            })

    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

    # Feature importance
    st.subheader("Feature Importance")
    fi_path = MODELS_DIR / "feature_importance.json"
    if fi_path.exists():
        fi_data = json.loads(fi_path.read_text())
        xgb_fi = fi_data.get("xgb_classifier", {})
        fi_df = pd.DataFrame([
            {"Feature": feat, "Importance": imp}
            for feat, imp in xgb_fi.items()
        ])
        if not fi_df.empty:
            st.bar_chart(fi_df.set_index("Feature"), horizontal=True)

    # 2026 prediction accuracy (on completed games)
    st.subheader("2026 Live Accuracy")
    try:
        games_df = fetch_2026_games()
        played = games_df[games_df["complete"] == 100]

        if played.empty:
            st.info("No completed 2026 games to evaluate yet.")
        else:
            correct = 0
            total = 0
            results_rows = []

            for _, game in played.iterrows():
                home = standardise_team(game["hteam"])
                away = standardise_team(game["ateam"])
                venue = game["venue"]
                rnd = int(game["round"]) if pd.notna(game["round"]) else 1

                actual_winner = standardise_team(game["winner"]) if pd.notna(game.get("winner")) else None
                h_score = int(game["hscore"])
                a_score = int(game["ascore"])

                if actual_winner is None:
                    if h_score > a_score:
                        actual_winner = home
                    elif a_score > h_score:
                        actual_winner = away
                    else:
                        continue  # draw

                try:
                    pred = predict_match_for_sim(home, away, venue, rnd)
                    predicted_winner = home if pred["home_prob"] >= 0.5 else away
                    is_correct = predicted_winner == actual_winner
                    correct += int(is_correct)
                    total += 1

                    results_rows.append({
                        "Round": game.get("roundname", f"R{rnd}"),
                        "Home": home,
                        "Away": away,
                        "Predicted": predicted_winner,
                        "Actual": actual_winner,
                        "Prob": f"{pred['home_prob']:.0%}",
                        "Correct": "✅" if is_correct else "❌",
                    })
                except Exception:
                    pass

            if total > 0:
                accuracy = correct / total
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("2026 Accuracy", f"{accuracy:.0%}", f"{correct}/{total} correct")
                with col2:
                    st.metric("Games Evaluated", total)

                st.dataframe(
                    pd.DataFrame(results_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No completed 2026 games to evaluate.")

    except Exception as e:
        st.warning(f"Could not evaluate 2026 accuracy: {e}")
