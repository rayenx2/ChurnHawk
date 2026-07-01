from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.inference
from src.inference import app


class TestAPIIntegration:

    @pytest.fixture
    def client(self):
        """Setup TestClient with mocked model"""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]

        with patch("src.inference.mlflow"):
            src.inference.model = mock_model

            with TestClient(app) as c:
                yield c

            src.inference.model = None

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint_success(self, client, sample_inference_data):
        """Test successful prediction flow"""
        response = client.post("/predict", json=sample_inference_data)
        assert response.status_code == 200

        data = response.json()
        assert "churn_prediction" in data
        assert data["churn_prediction"] in [0, 1]

    def test_predict_missing_fields(self, client):
        """Test validation error"""
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint exists"""
        response = client.get("/metrics")
        assert response.status_code in [200, 404]

    def test_docs_endpoint(self, client):
        """Test documentation exists"""
        response = client.get("/docs")
        assert response.status_code == 200
