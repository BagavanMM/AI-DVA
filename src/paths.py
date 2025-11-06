# paths.py
# Centralized path configuration for the project

import os
from pathlib import Path

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Common data files
ALL_BUGS_PROCESSED = DATA_DIR / "all_bugs_processed.json"
DATA_TRAIN = DATA_DIR / "data_train.json"
DATA_VAL = DATA_DIR / "data_val.json"
DATA_TEST = DATA_DIR / "data_test.json"
FEEDBACK_HISTORY = DATA_DIR / "feedback_history.json"
SAMPLE_BUGS_WITH_CODE = DATA_DIR / "sample_bugs_with_code.json"

# Model files
BUG_CLASSIFIER = MODELS_DIR / "bug_classifier.pkl"
BUG_DATABASE = MODELS_DIR / "bug_database.pkl"
SIMILARITY_INDEX = MODELS_DIR / "similarity_index.faiss"
VOCABULARY = MODELS_DIR / "vocabulary.pkl"
FEATURES_TRAIN = MODELS_DIR / "features_train.pkl"
FEATURES_VAL = MODELS_DIR / "features_val.pkl"
FEATURES_TEST = MODELS_DIR / "features_test.pkl"

# Convert to strings for compatibility
def get_data_path(filename: str) -> str:
    """Get path to a file in data directory"""
    return str(DATA_DIR / filename)

def get_model_path(filename: str) -> str:
    """Get path to a file in models directory"""
    return str(MODELS_DIR / filename)
