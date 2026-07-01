from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.inference
from src.inference import app


class TestInferenceAPI:
    @pytest.fixture
    def client(self):
        """Fixture to provide a TestClient with a mocked model"""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]

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
        """Test successful prediction"""
        response = client.post("/predict", json=sample_inference_data)

        assert response.status_code == 200
        result = response.json()
        assert "churn_prediction" in result
        assert result["churn_prediction"] == 1

    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction with invalid data"""
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_predict_endpoint_missing_fields(self, client, sample_inference_data):
        """Test prediction with missing required fields"""
        data = sample_inference_data.copy()
        del data["Contract"]
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            assert "model_name" in response.json()

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
