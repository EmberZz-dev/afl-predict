"""AFL match data cleaning and standardisation.

Reads the raw matches CSV produced by :mod:`src.data.collect`, applies
cleaning transformations, and writes a processed CSV ready for feature
engineering.

Usage:
    python -m src.data.clean
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "matches.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"

# Mapping of common alternative team names to their canonical form.
# Keys are lowercased for case-insensitive matching.
TEAM_NAME_MAP: dict[str, str] = {
    "gws giants": "Greater Western Sydney",
    "gws": "Greater Western Sydney",
    "greater western sydney giants": "Greater Western Sydney",
    "brisbane lions": "Brisbane",
    "brisbane bears": "Brisbane",
    "sydney swans": "Sydney",
    "geelong cats": "Geelong",
    "west coast eagles": "West Coast",
    "adelaide crows": "Adelaide",
    "western bulldogs": "Western Bulldogs",
    "footscray": "Western Bulldogs",
    "gold coast suns": "Gold Coast",
    "kangaroos": "North Melbourne",
    "north melbourne kangaroos": "North Melbourne",
    "melbourne demons": "Melbourne",
    "port adelaide power": "Port Adelaide",
    "richmond tigers": "Richmond",
    "carlton blues": "Carlton",
    "collingwood magpies": "Collingwood",
    "essendon bombers": "Essendon",
    "fremantle dockers": "Fremantle",
    "hawthorn hawks": "Hawthorn",
    "st kilda saints": "St Kilda",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def standardise_team_name(name: str) -> str:
    """Return the canonical team name for *name*.

    Performs a case-insensitive lookup against ``TEAM_NAME_MAP``.  If no
    mapping is found the original name is returned with surrounding
    whitespace stripped.
    """
    if not isinstance(name, str):
        return name
    cleaned = name.strip()
    return TEAM_NAME_MAP.get(cleaned.lower(), cleaned)


def _extract_round_number(value: object) -> int | None:
    """Extract a numeric round number from a round string or number.

    Handles values like ``"1"``, ``1``, ``"Round 5"``, or ``"QF"`` (returns
    ``None`` for non-numeric finals identifiers).
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    # Pure integer
    try:
        return int(s)
    except ValueError:
        pass
    # "Round 5" style
    match = re.search(r"(\d+)", s)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------


def clean_data(
    raw_path: Path = RAW_PATH,
    output_path: Path = CLEAN_PATH,
) -> pd.DataFrame:
    """Load raw match data, clean it, and save a processed CSV.

    Steps performed:
      1. Lowercase / snake_case column names.
      2. Drop rows missing critical fields (teams or scores).
      3. Standardise team names.
      4. Parse dates.
      5. Derive new columns: ``margin``, ``winner``, ``is_home_win``,
         ``season``, ``round_number``.
      6. Save to *output_path*.

    Args:
        raw_path: Path to the raw matches CSV.
        output_path: Destination for the cleaned CSV.

    Returns:
        The cleaned :class:`~pandas.DataFrame`.
    """
    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # 1. Normalise column names ------------------------------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # 2. Drop rows with missing essential fields --------------------------------
    essential = ["hteam", "ateam", "hscore", "ascore"]
    missing_before = len(df)
    df = df.dropna(subset=[c for c in essential if c in df.columns])
    dropped = missing_before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing essential fields", dropped)

    # Ensure scores are numeric
    for col in ("hscore", "ascore"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["hscore", "ascore"])

    # 3. Standardise team names -------------------------------------------------
    for col in ("hteam", "ateam", "winner"):
        if col in df.columns:
            df[col] = df[col].apply(standardise_team_name)

    # 4. Parse dates ------------------------------------------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 5. Derived columns --------------------------------------------------------
    df["margin"] = (df["hscore"] - df["ascore"]).astype(int)

    # Determine winner from scores (overwrite API field for consistency).
    df["winner"] = df.apply(
        lambda row: (
            row["hteam"]
            if row["hscore"] > row["ascore"]
            else (row["ateam"] if row["ascore"] > row["hscore"] else "Draw")
        ),
        axis=1,
    )

    df["is_home_win"] = (df["hscore"] > df["ascore"]).astype(int)

    # Season — prefer the explicit 'year' column, fall back to date.
    if "year" in df.columns:
        df["season"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    elif "date" in df.columns:
        df["season"] = df["date"].dt.year
    else:
        df["season"] = pd.NA

    # Round number — extract from 'round' or 'roundname'.
    round_source = "round" if "round" in df.columns else "roundname"
    if round_source in df.columns:
        df["round_number"] = df[round_source].apply(_extract_round_number).astype("Int64")
    else:
        df["round_number"] = pd.NA

    # 6. Save -------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d cleaned rows to %s", len(df), output_path)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting data cleaning…")

    try:
        df = clean_data()
        logger.info("Done. Shape: %s", df.shape)
    except FileNotFoundError:
        logger.error(
            "Raw data not found at %s. Run `python -m src.data.collect` first.",
            RAW_PATH,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
