import random
import time

import pytest
from fastapi.testclient import TestClient

from src.inference import app

client = TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_load_simulation(sample_payload):
    """Simulate load test"""
    start_time = time.time()
    requests_count = 50

    for _ in range(requests_count):
        # Vary data slightly
        payload = sample_payload.copy()
        payload["tenure"] = random.randint(1, 72)
        payload["MonthlyCharges"] = random.uniform(20.0, 100.0)

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    duration = time.time() - start_time
    assert duration < 5.0


def test_string_formatting_fixes():
    """Test placeholder to fix F541 errors"""
    # Flake8 complained about f-strings with no placeholders
    # We remove the 'f' prefix here
    msg = "This is a normal string now"
    assert isinstance(msg, str)

    msg2 = "Another normal string"
    assert len(msg2) > 0

    msg3 = "Final check"
    assert "check" in msg3
