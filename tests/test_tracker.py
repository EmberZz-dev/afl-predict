"""Tests for prediction monitoring and drift detection."""

import csv
import tempfile
from pathlib import Path

import pytest

from src.monitoring.tracker import PredictionTracker


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with a temporary log file."""
    log_path = tmp_path / "test_log.csv"
    return PredictionTracker(log_path=log_path)


class TestPredictionLogging:
    def test_log_creates_file(self, tracker) -> None:
        tracker.log_prediction(
            home_team="Carlton", away_team="Richmond",
            venue="MCG", round_number=5,
            predicted_winner="Carlton", home_win_prob=0.65,
        )
        assert tracker.log_path.exists()

    def test_log_appends_rows(self, tracker) -> None:
        for i in range(3):
            tracker.log_prediction(
                home_team="Carlton", away_team="Richmond",
                venue="MCG", round_number=i + 1,
                predicted_winner="Carlton", home_win_prob=0.6,
            )
        rows = tracker._read_all_rows()
        assert len(rows) == 3

    def test_log_with_actual_result(self, tracker) -> None:
        tracker.log_prediction(
            home_team="Carlton", away_team="Richmond",
            venue="MCG", round_number=5,
            predicted_winner="Carlton", home_win_prob=0.65,
            actual_winner="Carlton",
        )
        rows = tracker._read_all_rows()
        assert rows[0]["correct"] == "True"

    def test_log_wrong_prediction(self, tracker) -> None:
        tracker.log_prediction(
            home_team="Carlton", away_team="Richmond",
            venue="MCG", round_number=5,
            predicted_winner="Carlton", home_win_prob=0.65,
            actual_winner="Richmond",
        )
        rows = tracker._read_all_rows()
        assert rows[0]["correct"] == "False"


class TestRecordActualResult:
    def test_backfill_actual_winner(self, tracker) -> None:
        tracker.log_prediction(
            home_team="Carlton", away_team="Richmond",
            venue="MCG", round_number=5,
            predicted_winner="Carlton", home_win_prob=0.65,
        )
        updated = tracker.record_actual_result(
            home_team="Carlton", away_team="Richmond",
            round_number=5, actual_winner="Carlton",
        )
        assert updated == 1
        rows = tracker._read_all_rows()
        assert rows[0]["actual_winner"] == "Carlton"
        assert rows[0]["correct"] == "True"

    def test_no_match_returns_zero(self, tracker) -> None:
        tracker.log_prediction(
            home_team="Carlton", away_team="Richmond",
            venue="MCG", round_number=5,
            predicted_winner="Carlton", home_win_prob=0.65,
        )
        updated = tracker.record_actual_result(
            home_team="Geelong", away_team="Sydney",
            round_number=5, actual_winner="Geelong",
        )
        assert updated == 0


class TestAccuracyReport:
    def test_empty_report(self, tracker) -> None:
        report = tracker.get_report()
        assert report["total_predictions"] == 0
        assert report["all_time"] is None
        assert report["drift_detected"] is False

    def test_report_with_predictions(self, tracker) -> None:
        # Log 10 correct, 5 wrong
        for i in range(10):
            tracker.log_prediction(
                home_team="Carlton", away_team="Richmond",
                venue="MCG", round_number=i + 1,
                predicted_winner="Carlton", home_win_prob=0.65,
                actual_winner="Carlton",
            )
        for i in range(5):
            tracker.log_prediction(
                home_team="Sydney", away_team="Geelong",
                venue="SCG", round_number=i + 1,
                predicted_winner="Sydney", home_win_prob=0.6,
                actual_winner="Geelong",
            )
        report = tracker.get_report()
        assert report["total_predictions"] == 15
        assert abs(report["all_time"] - 0.6667) < 0.01

    def test_drift_detected_when_accuracy_low(self, tracker) -> None:
        # Log 20 mostly wrong predictions
        for i in range(20):
            winner = "Carlton" if i < 5 else "Richmond"
            tracker.log_prediction(
                home_team="Carlton", away_team="Richmond",
                venue="MCG", round_number=i + 1,
                predicted_winner="Carlton", home_win_prob=0.65,
                actual_winner=winner,
            )
        report = tracker.get_report()
        assert report["drift_detected"] is True  # 5/20 = 25% < 55%

    def test_no_drift_when_accuracy_high(self, tracker) -> None:
        for i in range(20):
            tracker.log_prediction(
                home_team="Carlton", away_team="Richmond",
                venue="MCG", round_number=i + 1,
                predicted_winner="Carlton", home_win_prob=0.65,
                actual_winner="Carlton",
            )
        report = tracker.get_report()
        assert report["drift_detected"] is False
