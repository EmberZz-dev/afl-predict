"""Season simulation engine for AFL ladder prediction.

Simulates the remaining 2026 season using trained models to predict
each unplayed match, then ranks teams by premiership points and percentage.
Supports Monte Carlo simulation for probabilistic ladder positions.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import (
    ELO_HOME_ADV,
    ELO_K,
    ELO_REGRESSION,
    ELO_START,
    _elo_expected,
    _elo_update,
    _regress_elo,
)


def build_ladder(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an AFL ladder from a list of match results.

    Parameters
    ----------
    results:
        List of dicts with keys: home_team, away_team, home_score, away_score.

    Returns
    -------
    pd.DataFrame
        Ladder sorted by points (desc), then percentage (desc).
    """
    teams: dict[str, dict[str, float]] = {}

    for match in results:
        home = match["home_team"]
        away = match["away_team"]
        h_score = match["home_score"]
        a_score = match["away_score"]

        for team in (home, away):
            if team not in teams:
                teams[team] = {
                    "played": 0, "wins": 0, "losses": 0, "draws": 0,
                    "points_for": 0, "points_against": 0, "premiership_points": 0,
                }

        teams[home]["played"] += 1
        teams[away]["played"] += 1
        teams[home]["points_for"] += h_score
        teams[home]["points_against"] += a_score
        teams[away]["points_for"] += a_score
        teams[away]["points_against"] += h_score

        if h_score > a_score:
            teams[home]["wins"] += 1
            teams[home]["premiership_points"] += 4
            teams[away]["losses"] += 1
        elif a_score > h_score:
            teams[away]["wins"] += 1
            teams[away]["premiership_points"] += 4
            teams[home]["losses"] += 1
        else:
            teams[home]["draws"] += 1
            teams[away]["draws"] += 1
            teams[home]["premiership_points"] += 2
            teams[away]["premiership_points"] += 2

    rows = []
    for team, stats in teams.items():
        pf = stats["points_for"]
        pa = stats["points_against"]
        percentage = (pf / pa * 100) if pa > 0 else 0.0
        rows.append({
            "team": team,
            "played": int(stats["played"]),
            "wins": int(stats["wins"]),
            "losses": int(stats["losses"]),
            "draws": int(stats["draws"]),
            "points_for": int(pf),
            "points_against": int(pa),
            "percentage": round(percentage, 1),
            "pts": int(stats["premiership_points"]),
        })

    ladder = pd.DataFrame(rows)
    ladder = ladder.sort_values(
        ["pts", "percentage"], ascending=[False, False]
    ).reset_index(drop=True)
    ladder.index = ladder.index + 1  # 1-based rank
    ladder.index.name = "rank"
    return ladder


def simulate_match(
    home_prob: float,
    home_elo: float,
    away_elo: float,
    stochastic: bool = True,
) -> dict[str, Any]:
    """Simulate a single match outcome.

    Parameters
    ----------
    home_prob:
        Model's predicted probability of home win.
    home_elo, away_elo:
        Current ELO ratings (used for margin estimation).
    stochastic:
        If True, randomly determine winner based on probability.
        If False, deterministically pick the more likely winner.

    Returns
    -------
    dict with home_win (bool), home_score (int), away_score (int).
    """
    if stochastic:
        home_win = random.random() < home_prob
    else:
        home_win = home_prob >= 0.5

    # Estimate margin from ELO difference (roughly 1 ELO point ≈ 0.15 margin points)
    elo_diff = home_elo - away_elo + ELO_HOME_ADV
    expected_margin = elo_diff * 0.15

    if stochastic:
        # Add noise: AFL margins have std dev ~35 points
        noise = random.gauss(0, 35)
        margin = expected_margin + noise
        # Ensure margin direction matches the winner
        if home_win and margin < 1:
            margin = abs(margin) + random.uniform(1, 10)
        elif not home_win and margin > -1:
            margin = -abs(margin) - random.uniform(1, 10)
    else:
        margin = expected_margin if home_win else -abs(expected_margin)

    margin = round(margin)
    avg_total = 170  # average combined score in AFL
    if home_win:
        home_score = max(30, int((avg_total + abs(margin)) / 2))
        away_score = max(20, int((avg_total - abs(margin)) / 2))
    else:
        away_score = max(30, int((avg_total + abs(margin)) / 2))
        home_score = max(20, int((avg_total - abs(margin)) / 2))

    return {
        "home_win": home_win,
        "home_score": home_score,
        "away_score": away_score,
        "margin": home_score - away_score,
    }


def simulate_season(
    played_results: list[dict[str, Any]],
    upcoming_matches: list[dict[str, Any]],
    predict_fn,
    n_simulations: int = 1000,
    stochastic: bool = True,
) -> dict[str, Any]:
    """Simulate the full season multiple times.

    Parameters
    ----------
    played_results:
        Completed match results with home_team, away_team, home_score, away_score.
    upcoming_matches:
        Unplayed matches with home_team, away_team, venue, round_number.
    predict_fn:
        Callable(home_team, away_team, venue, round_number) -> dict with
        home_prob, home_elo, away_elo.
    n_simulations:
        Number of Monte Carlo simulations (only used if stochastic=True).
    stochastic:
        If True, run Monte Carlo. If False, run one deterministic simulation.

    Returns
    -------
    dict with:
        deterministic_ladder: DataFrame of the most-likely ladder
        position_probabilities: DataFrame with average position per team
        finals_probability: dict of team -> probability of making top 8
        premiership_probability: dict of team -> probability of finishing 1st
    """
    n_runs = n_simulations if stochastic else 1

    # Track position counts across simulations
    all_teams = set()
    for r in played_results:
        all_teams.add(r["home_team"])
        all_teams.add(r["away_team"])
    for m in upcoming_matches:
        all_teams.add(m["home_team"])
        all_teams.add(m["away_team"])

    position_counts: dict[str, list[int]] = {
        team: [0] * 18 for team in all_teams
    }
    total_wins: dict[str, list[int]] = {team: [] for team in all_teams}

    # Pre-compute predictions for upcoming matches (probabilities don't change)
    predictions = []
    for match in upcoming_matches:
        pred = predict_fn(
            match["home_team"], match["away_team"],
            match["venue"], match["round_number"],
        )
        predictions.append(pred)

    for _ in range(n_runs):
        # Start with actual results
        sim_results = list(played_results)

        # Simulate remaining matches
        for match, pred in zip(upcoming_matches, predictions):
            outcome = simulate_match(
                home_prob=pred["home_prob"],
                home_elo=pred.get("home_elo", 1500),
                away_elo=pred.get("away_elo", 1500),
                stochastic=stochastic,
            )
            sim_results.append({
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "home_score": outcome["home_score"],
                "away_score": outcome["away_score"],
            })

        # Build ladder for this simulation
        ladder = build_ladder(sim_results)
        for pos, row in ladder.iterrows():
            team = row["team"]
            if team in position_counts:
                position_counts[team][pos - 1] += 1
                total_wins[team].append(row["wins"])

    # Aggregate results
    avg_position = {}
    finals_prob = {}
    premiership_prob = {}
    avg_wins_dict = {}

    for team in all_teams:
        counts = position_counts[team]
        total = sum(counts)
        if total == 0:
            continue
        avg_pos = sum((i + 1) * c for i, c in enumerate(counts)) / total
        avg_position[team] = round(avg_pos, 1)
        finals_prob[team] = round(sum(counts[:8]) / total * 100, 1)
        premiership_prob[team] = round(counts[0] / total * 100, 1)
        avg_wins_dict[team] = round(np.mean(total_wins[team]), 1)

    # Build deterministic ladder (single run with no randomness)
    det_results = list(played_results)
    for match, pred in zip(upcoming_matches, predictions):
        outcome = simulate_match(
            home_prob=pred["home_prob"],
            home_elo=pred.get("home_elo", 1500),
            away_elo=pred.get("away_elo", 1500),
            stochastic=False,
        )
        det_results.append({
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_score": outcome["home_score"],
            "away_score": outcome["away_score"],
        })
    deterministic_ladder = build_ladder(det_results)

    # Add simulation stats to deterministic ladder
    deterministic_ladder["avg_wins"] = deterministic_ladder["team"].map(avg_wins_dict)
    deterministic_ladder["finals_prob"] = deterministic_ladder["team"].map(finals_prob)
    deterministic_ladder["flag_prob"] = deterministic_ladder["team"].map(premiership_prob)

    return {
        "deterministic_ladder": deterministic_ladder,
        "finals_probability": finals_prob,
        "premiership_probability": premiership_prob,
        "avg_position": avg_position,
        "n_simulations": n_runs,
    }
