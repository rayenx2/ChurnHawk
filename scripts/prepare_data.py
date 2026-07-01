#!/usr/bin/env python3
"""
Data preparation script for DVC pipeline
Splits raw data into train and test sets
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings  # noqa: E402
from src.utils import dvc_add, ensure_dvc_data, load_params, save_json, setup_logging  # noqa: E402

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)


def calculate_data_stats(df):
    """Calculate statistics about the dataset"""
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "churn_rate": float(df["Churn"].value_counts(normalize=True).get("Yes", 0)),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
    }
    return stats


def prepare_data(input_path, output_dir, params=None):
    """
    Prepare and split data for training

    Args:
        input_path: Path to raw data CSV
        output_dir: Directory to save processed data
        params: Dictionary with preparation parameters
    """
    logger.info(f"Starting data preparation from {input_path}")

    # Ensure raw data is available
    if not ensure_dvc_data(input_path):
        logger.error(f"Failed to get raw data: {input_path}")
        raise FileNotFoundError(f"Raw data not found: {input_path}")

    # Load parameters
    if params is None:
        params = load_params().get("prepare", {})

    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    stratify = params.get("stratify", True)

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Calculate initial statistics
    initial_stats = calculate_data_stats(df)
    logger.info(f"Initial churn rate: {initial_stats['churn_rate']:.2%}")

    # Basic cleaning
    logger.info("Performing basic data cleaning...")

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(0, inplace=True)

    # Remove duplicates
    df = df.drop_duplicates()

    # Split data
    logger.info(f"Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")

    stratify_col = df["Churn"] if stratify else None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_col
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train data saved: {train_path} ({len(train_df)} rows)")
    logger.info(f"Test data saved: {test_path} ({len(test_df)} rows)")

    # Calculate final statistics
    train_stats = calculate_data_stats(train_df)
    test_stats = calculate_data_stats(test_df)

    stats = {
        "initial": initial_stats,
        "train": train_stats,
        "test": test_stats,
        "split_ratio": {
            "train": float(len(train_df) / len(df)),
            "test": float(len(test_df) / len(df)),
        },
    }

    # Save statistics
    stats_path = output_path / "data_stats.json"
    save_json(stats, str(stats_path))
    logger.info(f"Statistics saved: {stats_path}")

    # Add to DVC tracking
    logger.info("Adding processed data to DVC...")
    dvc_add(str(train_path))
    dvc_add(str(test_path))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--input", type=str, default="data/raw/churn.csv", help="Path to raw data CSV file"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed", help="Directory to save processed data"
    )

    args = parser.parse_args()

    # Run preparation
    stats = prepare_data(input_path=args.input, output_dir=args.output)

    print("\nData preparation complete!")
    print(f"Train size: {stats['train']['total_rows']} rows")
    print(f"Test size: {stats['test']['total_rows']} rows")
    print(f"Train churn rate: {stats['train']['churn_rate']:.2%}")
    print(f"Test churn rate: {stats['test']['churn_rate']:.2%}")
