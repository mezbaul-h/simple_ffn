"""
Settings Module

This module defines various constants and paths used in the project.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT_DIR = PROJECT_ROOT / "data"
DATASET_FILE_PREFIX = "ce889_dataCollection"
DEFAULT_CHECKPOINT_FILENAME = "ffn_checkpoint_2.json"
