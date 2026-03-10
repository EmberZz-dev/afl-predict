"""Tests for data cleaning module."""

import pandas as pd
import pytest

from src.data.clean import standardise_team_name, _extract_round_number


class TestStandardiseTeamName:
    def test_known_alias(self) -> None:
        assert standardise_team_name("GWS Giants") == "Greater Western Sydney"

    def test_case_insensitive(self) -> None:
        assert standardise_team_name("gws giants") == "Greater Western Sydney"

    def test_brisbane_lions(self) -> None:
        assert standardise_team_name("Brisbane Lions") == "Brisbane"

    def test_already_canonical(self) -> None:
        assert standardise_team_name("Carlton") == "Carlton"

    def test_strips_whitespace(self) -> None:
        assert standardise_team_name("  Carlton  ") == "Carlton"

    def test_non_string_passthrough(self) -> None:
        assert standardise_team_name(None) is None

    def test_footscray_maps_to_bulldogs(self) -> None:
        assert standardise_team_name("Footscray") == "Western Bulldogs"

    def test_kangaroos_maps_to_north(self) -> None:
        assert standardise_team_name("Kangaroos") == "North Melbourne"


class TestExtractRoundNumber:
    def test_integer(self) -> None:
        assert _extract_round_number(5) == 5

    def test_string_integer(self) -> None:
        assert _extract_round_number("5") == 5

    def test_round_prefix(self) -> None:
        assert _extract_round_number("Round 12") == 12

    def test_non_numeric_returns_none(self) -> None:
        assert _extract_round_number("QF") is None

    def test_nan_returns_none(self) -> None:
        assert _extract_round_number(float("nan")) is None
