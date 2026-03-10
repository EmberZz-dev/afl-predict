"""Feature engineering pipeline for AFL match prediction.

Builds a feature matrix from cleaned match data. All features are computed
from historical data only (no data leakage). The pipeline computes ELO
ratings, rolling form stats, head-to-head records, venue stats, and
seasonal features.

Expected input columns in matches_clean.csv:
    date, season, round_number, home_team, away_team, venue,
    home_score, away_score, home_ground

Usage:
    python -m src.features.build
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"

# ── ELO constants ────────────────────────────────────────────────────────
ELO_K: int = 30
ELO_START: float = 1500.0
ELO_HOME_ADV: float = 40.0
ELO_REGRESSION: float = 0.20  # 20 % regression toward mean between seasons


# ── Helper functions ─────────────────────────────────────────────────────

def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Return expected win probability for player A given both ratings."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _elo_update(
    rating: float, expected: float, actual: float, k: int = ELO_K
) -> float:
    """Return the updated ELO rating after a single game."""
    return rating + k * (actual - expected)


def _regress_elo(rating: float, factor: float = ELO_REGRESSION) -> float:
    """Regress a rating toward ELO_START by *factor* (0‑1)."""
    return rating + factor * (ELO_START - rating)


def _streak_update(current_streak: int, won: bool) -> int:
    """Update a win/loss streak counter.

    Positive values = consecutive wins, negative = consecutive losses.
    """
    if won:
        return current_streak + 1 if current_streak > 0 else 1
    return current_streak - 1 if current_streak < 0 else -1


# ── Main builder ─────────────────────────────────────────────────────────

def build_features(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Build the full feature matrix from cleaned match data.

    Parameters
    ----------
    input_path:
        Path to the cleaned matches CSV. Defaults to the project-standard
        location ``data/processed/matches_clean.csv``.

    Returns
    -------
    pd.DataFrame
        One row per match, containing all engineered features plus target
        variables (``home_win``, ``margin``).
    """
    df = pd.read_csv(input_path, parse_dates=["date"])

    # Map column names from clean data to expected names
    column_map = {
        "hteam": "home_team",
        "ateam": "away_team",
        "hscore": "home_score",
        "ascore": "away_score",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Ensure required columns exist
    if "home_score" not in df.columns and "hscore" in df.columns:
        df["home_score"] = df["hscore"]
    if "away_score" not in df.columns and "ascore" in df.columns:
        df["away_score"] = df["ascore"]

    df = df.sort_values(["date", "round_number"]).reset_index(drop=True)

    # ── Accumulators (keyed by team name) ────────────────────────────
    elo: dict[str, float] = defaultdict(lambda: ELO_START)

    # Per-team history: list of dicts with keys the team cares about
    team_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Head-to-head history: key = frozenset({team_a, team_b})
    h2h_history: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)

    # Venue history: key = (team, venue)
    venue_history: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    # Streaks
    streaks: dict[str, int] = defaultdict(int)

    # Last-game date per team (for rest-days calculation)
    last_game_date: dict[str, pd.Timestamp] = {}

    # Track current season for ELO regression
    current_season: int | None = None

    # ── Output rows ──────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []

    for _, match in df.iterrows():
        home: str = match["home_team"]
        away: str = match["away_team"]
        venue: str = match["venue"]
        season: int = int(match["season"])
        match_date: pd.Timestamp = match["date"]

        # Season rollover: regress ELO ratings toward the mean
        if current_season is not None and season != current_season:
            for team in list(elo.keys()):
                elo[team] = _regress_elo(elo[team])
        current_season = season

        # ── Compute features BEFORE updating state ───────────────

        # ELO
        home_elo = elo[home]
        away_elo = elo[away]
        elo_diff = home_elo - away_elo + ELO_HOME_ADV
        elo_prob = _elo_expected(home_elo + ELO_HOME_ADV, away_elo)

        # Form (rolling stats)
        home_hist = team_history[home]
        away_hist = team_history[away]

        home_form_5 = _win_rate(home_hist, 5)
        home_form_10 = _win_rate(home_hist, 10)
        away_form_5 = _win_rate(away_hist, 5)
        away_form_10 = _win_rate(away_hist, 10)

        home_avg_margin_5 = _avg_stat(home_hist, "margin", 5)
        away_avg_margin_5 = _avg_stat(away_hist, "margin", 5)
        home_avg_score_5 = _avg_stat(home_hist, "score", 5)
        away_avg_score_5 = _avg_stat(away_hist, "score", 5)
        home_avg_conceded_5 = _avg_stat(home_hist, "conceded", 5)
        away_avg_conceded_5 = _avg_stat(away_hist, "conceded", 5)

        home_streak = streaks[home]
        away_streak = streaks[away]

        # Head-to-head (last 5 meetings, from home team's perspective)
        h2h_key = frozenset({home, away})
        h2h_list = h2h_history[h2h_key]
        h2h_home_wins_5 = _h2h_wins(h2h_list, home, 5)
        h2h_avg_margin_5 = _h2h_margin(h2h_list, home, 5)

        # Venue (last 3 years of games at this venue)
        three_years_ago = match_date - pd.DateOffset(years=3)
        home_venue_win_rate = _venue_win_rate(
            venue_history[(home, venue)], three_years_ago
        )
        away_venue_win_rate = _venue_win_rate(
            venue_history[(away, venue)], three_years_ago
        )
        is_home_ground = int(match.get("home_ground", 0)) if "home_ground" in match.index else 0

        # Rest days
        days_rest_home = _days_rest(last_game_date.get(home), match_date)
        days_rest_away = _days_rest(last_game_date.get(away), match_date)

        # Round number (straight from data)
        round_number = match["round_number"]

        # ── Targets ──────────────────────────────────────────────
        home_score: int = int(match["home_score"])
        away_score: int = int(match["away_score"])
        margin = home_score - away_score
        home_win = int(margin > 0)

        # ── Store row ────────────────────────────────────────────
        rows.append(
            {
                "date": match_date,
                "season": season,
                "round_number": round_number,
                "home_team": home,
                "away_team": away,
                "venue": venue,
                # ELO
                "home_elo": round(home_elo, 2),
                "away_elo": round(away_elo, 2),
                "elo_diff": round(elo_diff, 2),
                "elo_prob": round(elo_prob, 4),
                # Form
                "home_form_5": home_form_5,
                "home_form_10": home_form_10,
                "away_form_5": away_form_5,
                "away_form_10": away_form_10,
                "home_avg_margin_5": home_avg_margin_5,
                "away_avg_margin_5": away_avg_margin_5,
                "home_avg_score_5": home_avg_score_5,
                "away_avg_score_5": away_avg_score_5,
                "home_avg_conceded_5": home_avg_conceded_5,
                "away_avg_conceded_5": away_avg_conceded_5,
                "home_streak": home_streak,
                "away_streak": away_streak,
                # Head-to-head
                "h2h_home_wins_5": h2h_home_wins_5,
                "h2h_avg_margin_5": h2h_avg_margin_5,
                # Venue
                "home_venue_win_rate": home_venue_win_rate,
                "away_venue_win_rate": away_venue_win_rate,
                "is_home_ground": is_home_ground,
                # Seasonal
                "days_rest_home": days_rest_home,
                "days_rest_away": days_rest_away,
                # Targets
                "home_win": home_win,
                "margin": margin,
            }
        )

        # ── Update state AFTER recording features ────────────────
        home_won = margin > 0
        away_won = margin < 0

        # ELO update
        actual_home = 1.0 if home_won else (0.0 if away_won else 0.5)
        exp_home = _elo_expected(home_elo + ELO_HOME_ADV, away_elo)
        elo[home] = _elo_update(home_elo, exp_home, actual_home)
        elo[away] = _elo_update(away_elo, 1.0 - exp_home, 1.0 - actual_home)

        # Team histories
        team_history[home].append(
            {
                "won": home_won,
                "margin": margin,
                "score": home_score,
                "conceded": away_score,
            }
        )
        team_history[away].append(
            {
                "won": away_won,
                "margin": away_score - home_score,
                "score": away_score,
                "conceded": home_score,
            }
        )

        # H2H
        h2h_history[h2h_key].append(
            {
                "winner": home if home_won else (away if away_won else None),
                "home_team": home,
                "margin_for_home": margin,
            }
        )

        # Venue
        venue_history[(home, venue)].append(
            {"date": match_date, "won": home_won}
        )
        venue_history[(away, venue)].append(
            {"date": match_date, "won": away_won}
        )

        # Streaks
        streaks[home] = _streak_update(streaks[home], home_won)
        streaks[away] = _streak_update(streaks[away], away_won)

        # Last game date
        last_game_date[home] = match_date
        last_game_date[away] = match_date

    features = pd.DataFrame(rows)
    return features


# ── Rolling-stat helpers ─────────────────────────────────────────────────

def _win_rate(history: list[dict[str, Any]], n: int) -> float | None:
    """Win rate over the last *n* games, or None if insufficient history."""
    if len(history) < n:
        return None
    recent = history[-n:]
    return round(sum(1 for g in recent if g["won"]) / n, 4)


def _avg_stat(
    history: list[dict[str, Any]], key: str, n: int
) -> float | None:
    """Average of *key* over the last *n* games, or None if insufficient."""
    if len(history) < n:
        return None
    recent = history[-n:]
    return round(sum(g[key] for g in recent) / n, 2)


def _h2h_wins(
    meetings: list[dict[str, Any]], team: str, n: int
) -> int | None:
    """Count wins for *team* in the last *n* head-to-head meetings."""
    if len(meetings) < n:
        recent = meetings  # use whatever is available
    else:
        recent = meetings[-n:]
    if not recent:
        return None
    return sum(1 for m in recent if m["winner"] == team)


def _h2h_margin(
    meetings: list[dict[str, Any]], team: str, n: int
) -> float | None:
    """Average margin from *team*'s perspective over last *n* meetings."""
    if len(meetings) < n:
        recent = meetings
    else:
        recent = meetings[-n:]
    if not recent:
        return None
    margins: list[int] = []
    for m in recent:
        # margin_for_home is positive when home wins
        if m["home_team"] == team:
            margins.append(m["margin_for_home"])
        else:
            margins.append(-m["margin_for_home"])
    return round(sum(margins) / len(margins), 2)


def _venue_win_rate(
    games: list[dict[str, Any]], cutoff: pd.Timestamp
) -> float | None:
    """Win rate at a venue for games on or after *cutoff*."""
    recent = [g for g in games if g["date"] >= cutoff]
    if not recent:
        return None
    return round(sum(1 for g in recent if g["won"]) / len(recent), 4)


def _days_rest(
    last_date: pd.Timestamp | None, current_date: pd.Timestamp
) -> float | None:
    """Days between *last_date* and *current_date*, or None if unknown."""
    if last_date is None:
        return None
    return (current_date - last_date).days


# ── CLI entry point ──────────────────────────────────────────────────────

def main() -> None:
    """Build features and save to CSV."""
    print(f"Loading data from {INPUT_PATH}")
    features = build_features()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(features)} rows x {len(features.columns)} columns")
    print(f"  -> {OUTPUT_PATH}")
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"\nNull counts:\n{features.isnull().sum().to_string()}")


if __name__ == "__main__":
    main()
