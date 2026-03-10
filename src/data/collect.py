"""AFL match data collection from the Squiggle API.

Fetches historical AFL match results (2015-2025) from the free
Squiggle API (https://api.squiggle.com.au) and saves them as raw CSV.

Usage:
    python -m src.data.collect
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DATA_DIR / "matches.csv"

BASE_URL = "https://api.squiggle.com.au"
HEADERS = {"User-Agent": "AFL-Predict/1.0"}
RATE_LIMIT_SECONDS = 1.0

YEAR_START = 2015
YEAR_END = 2025

# Columns we keep from the API response (in output order).
MATCH_COLUMNS = [
    "id",
    "year",
    "round",
    "roundname",
    "date",
    "hteam",
    "hteamid",
    "hscore",
    "ateam",
    "ateamid",
    "ascore",
    "venue",
    "winner",
    "complete",
    "is_final",
    "is_grand_final",
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make a GET request to the Squiggle API with rate limiting.

    Args:
        endpoint: The API endpoint path (e.g. ``/``).
        params: Query parameters forwarded to ``requests.get``.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        requests.HTTPError: On non-2xx status codes.
    """
    url = f"{BASE_URL}{endpoint}"
    logger.debug("GET %s params=%s", url, params)

    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()

    # Respect rate limit
    time.sleep(RATE_LIMIT_SECONDS)

    return response.json()


def fetch_games_for_year(year: int) -> list[dict[str, Any]]:
    """Fetch all match results for a single AFL season.

    Args:
        year: The season year (e.g. 2023).

    Returns:
        A list of match dictionaries from the API.
    """
    data = _get("/", params={"q": "games", "year": year})
    games: list[dict[str, Any]] = data.get("games", [])
    logger.info("Year %d: fetched %d games", year, len(games))
    return games


def fetch_teams() -> list[dict[str, Any]]:
    """Fetch the current list of AFL teams.

    Returns:
        A list of team dictionaries from the API.
    """
    data = _get("/", params={"q": "teams"})
    teams: list[dict[str, Any]] = data.get("teams", [])
    logger.info("Fetched %d teams", len(teams))
    return teams


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_matches(matches: list[dict[str, Any]], path: Path) -> None:
    """Write match records to a CSV file.

    Only the columns listed in ``MATCH_COLUMNS`` are written. Missing
    keys are stored as empty strings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MATCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for match in matches:
            row = {col: match.get(col, "") for col in MATCH_COLUMNS}
            writer.writerow(row)

    logger.info("Saved %d matches to %s", len(matches), path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def collect_all(
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Collect AFL match data for the given year range and save to CSV.

    Args:
        year_start: First season to fetch (inclusive).
        year_end: Last season to fetch (inclusive).
        output_path: Destination CSV file.

    Returns:
        The path to the saved CSV file.
    """
    all_matches: list[dict[str, Any]] = []

    for year in range(year_start, year_end + 1):
        try:
            games = fetch_games_for_year(year)
            all_matches.extend(games)
        except requests.RequestException as exc:
            logger.error("Failed to fetch year %d: %s", year, exc)
            continue

    if not all_matches:
        logger.warning("No matches collected — nothing to save.")
        return output_path

    # Sort by date for deterministic output
    all_matches.sort(key=lambda g: (g.get("year", 0), g.get("round", 0), g.get("id", 0)))

    _save_matches(all_matches, output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting AFL data collection (%d–%d)…", YEAR_START, YEAR_END)

    try:
        path = collect_all()
        logger.info("Done. Output: %s", path)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
