import pandas as pd

from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:

    def test_feature_engineer_initialization(self):
        """Test feature engineer can be initialized"""
        fe = FeatureEngineer()
        assert fe is not None
        assert fe.label_encoders == {}

    def test_fit_transform(self, sample_data):
        """Test fit_transform creates encoded features"""
        fe = FeatureEngineer()
        df_transformed = fe.fit_transform(sample_data.drop("Churn", axis=1))

        # Check that categorical columns are encoded
        assert pd.api.types.is_numeric_dtype(df_transformed["gender"])
        assert pd.api.types.is_numeric_dtype(df_transformed["Contract"])

        # Check label encoders were fitted
        assert len(fe.label_encoders) > 0

    def test_transform_consistency(self, sample_data):
        """Test transform produces consistent results"""
        fe = FeatureEngineer()

        # Fit on full data
        fe.fit_transform(sample_data.drop("Churn", axis=1))

        # Transform subset
        subset = sample_data.iloc[:10].drop("Churn", axis=1)
        transformed = fe.transform(subset)

        assert len(transformed) == 10
        assert all(pd.api.types.is_numeric_dtype(transformed[col]) for col in transformed.columns)

    def test_new_feature_creation(self, sample_data):
        """Test creation of engineered features"""
        fe = FeatureEngineer()
        df_transformed = fe.fit_transform(sample_data.drop("Churn", axis=1))

        # Check new features exist
        assert "charges_per_month" in df_transformed.columns
        assert "tenure_group" in df_transformed.columns

    def test_handle_unseen_categories(self, sample_data):
        """Test handling of unseen categorical values"""
        fe = FeatureEngineer()
        train_data = sample_data[:80].drop("Churn", axis=1)
        test_data = sample_data[80:].drop("Churn", axis=1).copy()

        # Fit on train
        fe.fit_transform(train_data)

        # Add unseen category
        test_data.loc[test_data.index[0], "gender"] = "Unknown"

        # Should handle gracefully
        transformed = fe.transform(test_data)
        assert transformed is not None

    def test_feature_scaling(self, sample_data):
        """Test numerical features are properly scaled"""
        fe = FeatureEngineer(scale_features=True)
        df_transformed = fe.fit_transform(sample_data.drop("Churn", axis=1))

        # Check that numerical features are scaled
        assert df_transformed["MonthlyCharges"].mean() < 1
        assert df_transformed["tenure"].std() <= 1.5
