import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config  # type: ignore


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load DVC parameters from params.yaml"""
    return load_config(params_path)


def save_json(data: Dict[str, Any], path: str):
    """Save dictionary to JSON file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(path, "r") as f:
        return json.load(f)  # type: ignore


def dvc_pull(path: Optional[str] = None) -> bool:
    """
    Pull data from DVC remote storage

    Args:
        path: Specific path to pull (optional, pulls all if None)

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["dvc", "pull"]
        if path:
            cmd.append(path)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info(f"DVC pull successful: {path or 'all files'}")
            return True
        else:
            logging.error(f"DVC pull failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Error during DVC pull: {e}")
        return False


def dvc_push(path: Optional[str] = None) -> bool:
    """
    Push data to DVC remote storage

    Args:
        path: Specific path to push (optional, pushes all if None)

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["dvc", "push"]
        if path:
            cmd.append(path)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info(f"DVC push successful: {path or 'all files'}")
            return True
        else:
            logging.error(f"DVC push failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Error during DVC push: {e}")
        return False


def dvc_add(path: str) -> bool:
    """
    Add file to DVC tracking

    Args:
        path: Path to file or directory to track

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(["dvc", "add", path], capture_output=True, text=True)

        if result.returncode == 0:
            logging.info(f"DVC add successful: {path}")
            # Auto-commit .dvc file
            dvc_file = f"{path}.dvc"
            subprocess.run(["git", "add", dvc_file, ".gitignore"])
            return True
        else:
            logging.error(f"DVC add failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Error during DVC add: {e}")
        return False


def ensure_dvc_data(path: str) -> bool:
    """
    Ensure data is available locally, pull from DVC if needed

    Args:
        path: Path to data file/directory

    Returns:
        True if data is available, False otherwise
    """
    if Path(path).exists():
        logging.info(f"Data already available: {path}")
        return True

    logging.info(f"Data not found locally, pulling from DVC: {path}")
    return dvc_pull(path)
