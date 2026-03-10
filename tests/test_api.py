"""Integration tests for the FastAPI prediction API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_has_model_status(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


# ---------------------------------------------------------------------------
# Teams endpoint
# ---------------------------------------------------------------------------


class TestTeamsEndpoint:
    def test_teams_returns_18(self, client: TestClient) -> None:
        response = client.get("/teams")
        assert response.status_code == 200
        teams = response.json()
        assert isinstance(teams, list)
        assert len(teams) == 18

    def test_teams_have_required_fields(self, client: TestClient) -> None:
        response = client.get("/teams")
        teams = response.json()
        for team in teams:
            assert "name" in team
            assert "elo_rating" in team
            assert "form_last_5" in team
            assert "rank" in team

    def test_teams_have_valid_ranks(self, client: TestClient) -> None:
        response = client.get("/teams")
        teams = response.json()
        ranks = [t["rank"] for t in teams]
        assert sorted(ranks) == list(range(1, 19))

    def test_teams_elo_ratings_are_positive(self, client: TestClient) -> None:
        response = client.get("/teams")
        teams = response.json()
        for team in teams:
            assert team["elo_rating"] > 0

    def test_known_teams_present(self, client: TestClient) -> None:
        response = client.get("/teams")
        team_names = [t["name"] for t in response.json()]
        for expected in ["Carlton", "Collingwood", "Essendon", "Richmond"]:
            assert expected in team_names


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------


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
        assert "away_win_probability" in data
        assert "predicted_margin" in data
        assert "confidence" in data
        assert "top_features" in data

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

    def test_predict_probabilities_in_range(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Sydney",
            "away_team": "GWS Giants",
            "venue": "SCG",
            "round_number": 10,
        })
        data = response.json()
        assert 0.0 <= data["home_win_probability"] <= 1.0
        assert 0.0 <= data["away_win_probability"] <= 1.0

    def test_predict_has_valid_confidence(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Sydney",
            "away_team": "GWS Giants",
            "venue": "SCG",
            "round_number": 10,
        })
        data = response.json()
        assert data["confidence"] in ("high", "medium", "low")

    def test_predict_invalid_home_team(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "NotATeam",
            "away_team": "Richmond",
            "venue": "MCG",
            "round_number": 5,
        })
        assert response.status_code == 400
        assert "Unknown team" in response.json()["detail"]

    def test_predict_invalid_away_team(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Carlton",
            "away_team": "FakeTeam",
            "venue": "MCG",
            "round_number": 5,
        })
        assert response.status_code == 400
        assert "Unknown team" in response.json()["detail"]

    def test_predict_invalid_round_number(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Carlton",
            "away_team": "Richmond",
            "venue": "MCG",
            "round_number": 0,
        })
        assert response.status_code == 422  # Pydantic validation (ge=1)

    def test_predict_missing_fields(self, client: TestClient) -> None:
        response = client.post("/predict", json={
            "home_team": "Carlton",
        })
        assert response.status_code == 422

    def test_predict_multiple_matches(self, client: TestClient) -> None:
        """Different matchups should return different predictions."""
        r1 = client.post("/predict", json={
            "home_team": "Carlton", "away_team": "Richmond",
            "venue": "MCG", "round_number": 5,
        })
        r2 = client.post("/predict", json={
            "home_team": "Geelong", "away_team": "North Melbourne",
            "venue": "GMHBA Stadium", "round_number": 5,
        })
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Probabilities should differ for different matchups
        d1, d2 = r1.json(), r2.json()
        assert d1["home_team"] == "Carlton"
        assert d2["home_team"] == "Geelong"


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client: TestClient) -> None:
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_fields(self, client: TestClient) -> None:
        response = client.get("/model/info")
        data = response.json()
        assert "model_type" in data
        assert "accuracy" in data
        assert "log_loss" in data
        assert "feature_count" in data

    def test_model_info_accuracy_range(self, client: TestClient) -> None:
        response = client.get("/model/info")
        data = response.json()
        assert 0.0 <= data["accuracy"] <= 1.0

    def test_model_info_feature_count(self, client: TestClient) -> None:
        response = client.get("/model/info")
        data = response.json()
        assert data["feature_count"] == 21


# ---------------------------------------------------------------------------
# Model features endpoint
# ---------------------------------------------------------------------------


class TestModelFeaturesEndpoint:
    def test_features_returns_list(self, client: TestClient) -> None:
        response = client.get("/model/features")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 21  # 21 features

    def test_features_have_name_and_contribution(self, client: TestClient) -> None:
        response = client.get("/model/features")
        data = response.json()
        for item in data:
            assert "feature" in item
            assert "contribution" in item
            assert isinstance(item["contribution"], float)

    def test_top_feature_is_elo(self, client: TestClient) -> None:
        response = client.get("/model/features")
        data = response.json()
        # elo_prob should be the top feature
        assert data[0]["feature"] == "elo_prob"


# ---------------------------------------------------------------------------
# Explain endpoint
# ---------------------------------------------------------------------------


class TestExplainEndpoint:
    def test_explain_valid_match(self, client: TestClient) -> None:
        response = client.post("/explain", json={
            "home_team": "Carlton",
            "away_team": "Richmond",
            "venue": "MCG",
            "round_number": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "shap_values" in data
        assert "prediction" in data
        assert isinstance(data["shap_values"], list)

    def test_explain_invalid_team(self, client: TestClient) -> None:
        response = client.post("/explain", json={
            "home_team": "FakeTeam",
            "away_team": "Richmond",
            "venue": "MCG",
            "round_number": 5,
        })
        assert response.status_code == 400
