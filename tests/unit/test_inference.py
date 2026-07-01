import numpy as np
import pandas as pd
import pytest

from src.train import evaluate_model, save_model, train_model


class TestModelTraining:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = pd.DataFrame(
            {
                "customerID": [f"C{i:04d}" for i in range(100)],
                "gender": np.random.choice(["Male", "Female"], 100),
                "SeniorCitizen": np.random.choice([0, 1], 100),
                "Partner": np.random.choice(["Yes", "No"], 100),
                "Dependents": np.random.choice(["Yes", "No"], 100),
                "tenure": np.random.randint(0, 72, 100),
                "PhoneService": "Yes",
                "MultipleLines": "No",
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
                "MonthlyCharges": np.random.uniform(20, 120, 100),
                "TotalCharges": np.random.uniform(20, 5000, 100),
                "Churn": np.random.choice(["Yes", "No"], 100),
            }
        )
        return data

    def test_train_model_returns_model(self, sample_data):
        """Test that training returns a model object"""
        X = sample_data.drop(["Churn", "customerID"], axis=1)
        y = sample_data["Churn"].map({"Yes": 1, "No": 0})

        from sklearn.preprocessing import LabelEncoder

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        model, metrics = train_model(X, y, params={"max_depth": 3, "n_estimators": 10})

        assert model is not None
        assert hasattr(model, "predict")
        assert isinstance(metrics, dict)

    def test_model_predictions_valid_range(self, sample_data):
        """Test model predictions are in valid range [0, 1]"""
        X = sample_data.drop(["Churn", "customerID"], axis=1)
        y = sample_data["Churn"].map({"Yes": 1, "No": 0})

        from sklearn.preprocessing import LabelEncoder

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        model, _ = train_model(X, y)
        predictions = model.predict(X)

        assert np.all(np.isin(predictions, [0, 1]))

    def test_model_evaluate_metrics(self, sample_data):
        """Test evaluation returns expected metrics"""
        X = sample_data.drop(["Churn", "customerID"], axis=1)
        y = sample_data["Churn"].map({"Yes": 1, "No": 0})

        from sklearn.preprocessing import LabelEncoder

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        model, _ = train_model(X, y)

        metrics, _, _ = evaluate_model(model, X, y)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics

    def test_model_save_load(self, sample_data, temp_data_dir):
        """Test model can be saved and loaded"""
        X = sample_data.drop(["Churn", "customerID"], axis=1)
        y = sample_data["Churn"].map({"Yes": 1, "No": 0})

        from sklearn.preprocessing import LabelEncoder

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        model, _ = train_model(X, y)

        model_path = temp_data_dir / "model.pkl"
        save_model(model, model_path)

        assert model_path.exists()

        import joblib

        loaded_model = joblib.load(model_path)

        predictions = loaded_model.predict(X[:5])
        assert len(predictions) == 5
