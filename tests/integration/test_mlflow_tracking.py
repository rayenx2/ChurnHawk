from unittest.mock import MagicMock, patch

import pytest

try:
    import mlflow
except ImportError:
    mlflow = None


@pytest.mark.skipif(mlflow is None, reason="MLflow library is not installed")
class TestMLflowIntegration:

    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow to avoid needing a real server"""
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_params") as mock_log_params,
            patch("mlflow.log_metrics") as mock_log_metrics,
            patch("mlflow.sklearn.log_model") as mock_log_model,
            patch("mlflow.active_run") as mock_active_run,
        ):

            mock_run_instance = MagicMock()
            mock_run.return_value.__enter__.return_value = mock_run_instance
            mock_active_run.return_value.info.run_id = "test_run_id"

            yield {
                "log_params": mock_log_params,
                "log_metrics": mock_log_metrics,
                "log_model": mock_log_model,
            }

    def test_log_params_to_mlflow(self, mock_mlflow):
        """Test logging parameters to MLflow (Mocked)"""
        params = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 100}

        mlflow.log_params(params)

        mock_mlflow["log_params"].assert_called_with(params)

    def test_log_metrics_to_mlflow(self, mock_mlflow):
        """Test logging metrics to MLflow (Mocked)"""
        metrics = {"accuracy": 0.85, "roc_auc": 0.88}

        mlflow.log_metrics(metrics)

        mock_mlflow["log_metrics"].assert_called_with(metrics)

    def test_log_model_to_mlflow(self, sample_data, mock_mlflow):
        """Test logging model artifact to MLflow (Mocked)"""
        X = sample_data.drop(["Churn", "customerID"], axis=1)
        y = sample_data["Churn"].map({"Yes": 1, "No": 0})

        from sklearn.preprocessing import LabelEncoder

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=2)
        model.fit(X, y)

        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")

        mock_mlflow["log_model"].assert_called()
