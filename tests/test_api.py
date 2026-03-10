"""Tests for the FastAPI prediction API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_has_model_status(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data


class TestTeamsEndpoint:
    def test_teams_returns_list(self, client: TestClient) -> None:
        response = client.get("/teams")
        assert response.status_code == 200
        teams = response.json()
        assert isinstance(teams, list)
        assert len(teams) == 18  # 18 AFL teams

    def test_teams_have_required_fields(self, client: TestClient) -> None:
        response = client.get("/teams")
        teams = response.json()
        for team in teams:
            assert "name" in team
            assert "elo_rating" in team


class TestPredictEndpoint:
    def test_predict_valid_match(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Carlton",
            "away_team": "Richmond",
            "venue": "MCG",
            "round_number": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "home_win_probability" in data
        assert "predicted_margin" in data
        assert 0 <= data["home_win_probability"] <= 1

    def test_predict_probabilities_sum_to_one(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Collingwood",
            "away_team": "Essendon",
            "venue": "MCG",
            "round_number": 1,
        })
        data = response.json()
        total = data["home_win_probability"] + data["away_win_probability"]
        assert abs(total - 1.0) < 0.01

    def test_predict_has_confidence(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Sydney",
            "away_team": "GWS Giants",
            "venue": "SCG",
            "round_number": 10,
        })
        data = response.json()
        assert data["confidence"] in ("high", "medium", "low")


class TestModelInfoEndpoint:
    def test_model_info_returns(self, client: TestClient) -> None:
        response = client.get("/model/info")
        assert response.status_code == 200
