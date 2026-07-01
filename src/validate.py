import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

# Import modules
from src.config import settings
from src.monitoring import monitor
from src.utils import dvc_push, ensure_dvc_data, save_json, setup_logging

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)


class DataValidator:
    """Data validation and drift detection"""

    def __init__(self):
        self.required_columns = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]

    def validate_schema(self, df):
        """
        Validate dataframe schema
        """
        logger.info("Validating data schema...")

        missing_columns = [col for col in self.required_columns if col not in df.columns]

        result = {
            "is_valid": len(missing_columns) == 0,
            "missing_columns": missing_columns,
            "total_columns": len(df.columns),
            "required_columns": len(self.required_columns),
        }

        if not result["is_valid"]:
            logger.warning(f"Schema validation failed. Missing columns: {missing_columns}")
        else:
            logger.info("Schema validation passed")

        return result

    def check_missing_values(self, df):
        """
        Check for missing values
        """
        logger.info("Checking for missing values...")

        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()

        total_missing = sum(missing_dict.values())
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100

        if missing_dict:
            logger.warning(f"Found missing values: {missing_dict}")
        else:
            logger.info("No missing values found")

        return {
            "missing_by_column": missing_dict,
            "total_missing": int(total_missing),
            "missing_percentage": float(missing_percentage),
        }

    def detect_outliers(self, df, column, method="iqr", threshold=1.5):
        """
        Detect outliers in a numerical column
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return {"count": 0, "percentage": 0.0}

        # Ensure column is numeric before calculating quantiles
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.warning(f"Column {column} is not numeric, skipping outlier detection.")
            return {"count": 0, "percentage": 0.0}

        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))
        elif method == "zscore":
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_count = outliers.sum()
        outlier_percentage = (outlier_count / len(df)) * 100

        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers ({outlier_percentage:.2f}%) in {column}")

        return {"count": int(outlier_count), "percentage": float(outlier_percentage)}

    def detect_drift(self, reference_data, current_data, report_path):
        """
        Detect data drift using Evidently
        """
        logger.info("Detecting data drift...")

        # Create Evidently report
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        # Extract drift results
        report_dict = report.as_dict()

        # Get drift status
        drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]

        # Save report
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(report_path))

        logger.info(f"Drift report saved to {report_path}")

        if drift_detected:
            logger.warning("Data drift detected")
            monitor.data_drift_detected.set(1)
        else:
            logger.info("No significant data drift detected")
            monitor.data_drift_detected.set(0)

        return {"drift_detected": drift_detected, "report_path": str(report_path)}


def validate_data_quality(df):
    """
    Calculate data quality metrics
    """
    logger.info("Calculating data quality metrics...")

    # Completeness: percentage of non-null values
    total_cells = len(df) * len(df.columns)
    null_cells = df.isnull().sum().sum()
    completeness = 1 - (null_cells / total_cells)

    # Uniqueness: check for duplicate rows
    duplicates = df.duplicated().sum()
    uniqueness = 1 - (duplicates / len(df))

    # Overall quality score (weighted average)
    quality_score = (completeness * 0.7 + uniqueness * 0.3) * 100

    metrics = {
        "completeness": float(completeness),
        "uniqueness": float(uniqueness),
        "quality_score": float(quality_score),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(duplicates),
        "null_cells": int(null_cells),
    }

    logger.info(f"Quality metrics: Quality Score={quality_score:.2f}%")

    # Update monitoring
    monitor.data_quality_score.set(quality_score)

    return metrics


def run_validation(input_path, output_dir=None, reference_path=None):
    """
    Run complete validation pipeline
    """
    logger.info(f"Starting validation pipeline for {input_path}")

    # Ensure input data is available
    if not ensure_dvc_data(input_path):
        logger.error(f"Failed to get input data: {input_path}")
        raise FileNotFoundError(f"Input data not found: {input_path}")

    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Initialize validator
    validator = DataValidator()

    # Run validations
    results = {
        "schema": validator.validate_schema(df),
        "missing_values": validator.check_missing_values(df),
        "quality_metrics": validate_data_quality(df),
        "outliers": {},
    }

    # Check outliers for numerical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numerical_cols:
        if col in df.columns:
            results["outliers"][col] = validator.detect_outliers(df, col)

    # Drift detection (if reference data provided)
    if reference_path:
        if not ensure_dvc_data(reference_path):
            logger.warning(f"Reference data not found: {reference_path}")
        else:
            reference_df = pd.read_csv(reference_path)

            # Також конвертуємо TotalCharges для референсного датасету
            if "TotalCharges" in reference_df.columns:
                reference_df["TotalCharges"] = pd.to_numeric(
                    reference_df["TotalCharges"], errors="coerce"
                ).fillna(0)

            # Ensure same columns
            common_cols = list(set(df.columns) & set(reference_df.columns))

            if output_dir is None:
                output_dir = settings.DATA_REPORTS_PATH

            report_path = Path(output_dir) / "validation_report.html"

            drift_results = validator.detect_drift(
                reference_df[common_cols], df[common_cols], report_path
            )
            results["drift"] = drift_results

    # Overall validation
    results["is_valid"] = results["schema"]["is_valid"]

    # Save results
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = Path(output_dir) / "validation_metrics.json"
        save_json(results, str(metrics_path))
        logger.info(f"Validation metrics saved to {metrics_path}")

        # Push reports to DVC
        dvc_push(str(output_dir))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate data quality and detect drift")
    parser.add_argument("--input", type=str, required=True, help="Path to input data CSV file")
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save validation reports"
    )
    parser.add_argument(
        "--reference", type=str, default=None, help="Path to reference data for drift detection"
    )

    args = parser.parse_args()

    # Run validation
    results = run_validation(
        input_path=args.input, output_dir=args.output, reference_path=args.reference
    )

    if results["is_valid"]:
        print("Validation passed!")
    else:
        print("Validation failed!")
        print(f"Missing columns: {results['schema']['missing_columns']}")

    print(f"Quality Score: {results['quality_metrics']['quality_score']:.2f}%")
