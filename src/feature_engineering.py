from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer
    Handles encoding, scaling, and feature creation
    """

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.categorical_columns_: List[str] = []
        self.numerical_columns_: List[str] = []
        self.scaler_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders and scaler on training data"""
        X = X.copy()

        # 1. Fix TotalCharges (Text -> Numeric)
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

        # 2. Identify types
        self.categorical_columns_ = X.select_dtypes(include=["object"]).columns.tolist()
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()

        # 3. Fit LabelEncoders
        for col in self.categorical_columns_:
            le = LabelEncoder()
            # Тимчасове заповнення для навчання енкодера
            temp_col = X[col].fillna("missing").astype(str)
            le.fit(temp_col)
            self.label_encoders[col] = le

        # 4. Create engineered features
        X_transformed = self._create_features(X)

        # 5. Fit Scaler
        if self.scale_features:
            self.scaler = StandardScaler()
            self.scaler_features_ = X_transformed.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            self.scaler.fit(X_transformed[self.scaler_features_])

        self.feature_names_ = X_transformed.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders and scaler"""
        X = X.copy()

        # 1. Fix TotalCharges
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

        # 2. Apply Label Encoding
        for col in self.categorical_columns_:
            if col in X.columns:
                X[col] = X[col].fillna("missing").astype(str)
                le = self.label_encoders[col]
                # Safe transform: unknown values -> 0
                X[col] = X[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])

        # 3. Create Features
        X = self._create_features(X)

        # 4. Apply Scaler
        if self.scale_features and self.scaler is not None:
            valid_features = [f for f in self.scaler_features_ if f in X.columns]
            if valid_features:
                X[valid_features] = self.scaler.transform(X[valid_features])

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        X = X.copy()

        # Feature 1: Charges per month
        if "TotalCharges" in X.columns and "tenure" in X.columns:
            X["charges_per_month"] = X["TotalCharges"] / (X["tenure"] + 1)

        # Feature 2: Tenure groups
        if "tenure" in X.columns:
            X["tenure_group"] = pd.cut(
                X["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
            )
            if "tenure_group" in X.columns:
                X["tenure_group"] = X["tenure_group"].astype(str)
                if "tenure_group" not in self.label_encoders:
                    le = LabelEncoder()
                    le.fit(X["tenure_group"])
                    self.label_encoders["tenure_group"] = le

                le = self.label_encoders["tenure_group"]
                X["tenure_group"] = X["tenure_group"].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # Feature 3: Service combination
        if all(col in X.columns for col in ["PhoneService", "InternetService"]):
            X["has_multiple_services"] = (
                (X["PhoneService"] == "Yes") & (X["InternetService"] != "No")
            ).astype(int)

        # Feature 4: Payment reliability indicator
        if "Contract" in X.columns:
            X["long_term_contract"] = (X["Contract"].isin(["One year", "Two year"])).astype(int)

        return X

    def save(self, path: str):
        """Save feature engineer to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        """Load feature engineer from disk"""
        return joblib.load(path)  # type: ignore
