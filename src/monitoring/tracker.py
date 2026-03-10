"""
Prediction monitoring and drift detection.

Logs every prediction, tracks rolling accuracy, and flags
model drift when performance degrades below threshold.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MONITORING_LOG_PATH = PROJECT_ROOT / "data" / "monitoring_log.csv"

DRIFT_THRESHOLD = 0.55
CSV_FIELDS = [
    "timestamp",
    "home_team",
    "away_team",
    "venue",
    "round_number",
    "predicted_winner",
    "home_win_prob",
    "actual_winner",
    "correct",
]


class PredictionTracker:
    """Tracks predictions, calculates rolling accuracy, and detects model drift."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or MONITORING_LOG_PATH
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        home_team: str,
        away_team: str,
        venue: str,
        round_number: int,
        predicted_winner: str,
        home_win_prob: float,
        actual_winner: str | None = None,
    ) -> None:
        """Append a prediction record to the monitoring log."""
        correct: str | None = None
        if actual_winner is not None:
            correct = str(predicted_winner == actual_winner)

        row = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "round_number": round_number,
            "predicted_winner": predicted_winner,
            "home_win_prob": round(home_win_prob, 4),
            "actual_winner": actual_winner or "",
            "correct": correct or "",
        }

        with self.log_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writerow(row)

    def record_actual_result(
        self,
        home_team: str,
        away_team: str,
        round_number: int,
        actual_winner: str,
    ) -> int:
        """
        Back-fill the actual winner for matching predictions.

        Returns the number of rows updated.
        """
        rows = self._read_all_rows()
        updated = 0

        for row in rows:
            if (
                row["home_team"] == home_team
                and row["away_team"] == away_team
                and str(row["round_number"]) == str(round_number)
                and not row.get("actual_winner")
            ):
                row["actual_winner"] = actual_winner
                row["correct"] = str(row["predicted_winner"] == actual_winner)
                updated += 1

        if updated > 0:
            self._write_all_rows(rows)

        return updated

    def get_report(self) -> dict:
        """
        Return an accuracy report.

        Keys:
            rolling_20  — accuracy over last 20 resolved predictions (or None)
            rolling_50  — accuracy over last 50 resolved predictions (or None)
            all_time    — accuracy over all resolved predictions (or None)
            total_predictions — total number of logged predictions
            drift_detected — True if rolling_20 accuracy < DRIFT_THRESHOLD
        """
        rows = self._read_all_rows()
        total_predictions = len(rows)

        resolved = [r for r in rows if r.get("correct") in ("True", "False")]

        all_time = self._accuracy(resolved) if resolved else None
        rolling_20 = self._accuracy(resolved[-20:]) if len(resolved) >= 1 else None
        rolling_50 = self._accuracy(resolved[-50:]) if len(resolved) >= 1 else None

        drift_detected = False
        if rolling_20 is not None and rolling_20 < DRIFT_THRESHOLD:
            drift_detected = True

        return {
            "rolling_20": rolling_20,
            "rolling_50": rolling_50,
            "all_time": all_time,
            "total_predictions": total_predictions,
            "drift_detected": drift_detected,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_log_file(self) -> None:
        """Create the log file with headers if it does not exist."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()

    def _read_all_rows(self) -> list[dict]:
        """Read every row from the monitoring CSV."""
        if not self.log_path.exists():
            return []
        with self.log_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            return list(reader)

    def _write_all_rows(self, rows: list[dict]) -> None:
        """Overwrite the monitoring CSV with the supplied rows."""
        with self.log_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _accuracy(rows: list[dict]) -> float:
        """Calculate accuracy from a list of resolved prediction rows."""
        if not rows:
            return 0.0
        correct = sum(1 for r in rows if r.get("correct") == "True")
        return round(correct / len(rows), 4)
