"""
config.py

Configuration settings for the Financial and Credit Risk Model Development project.
This module contains global parameters and settings that are used across other modules.
"""

# Data paths
DATA_FILE_PATH = "data/credit_data.csv"  # Path to the credit risk dataset

# Model hyperparameters for Random Forest
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}

# Preprocessing settings
MISSING_VALUE_STRATEGY = "mean"  # Options: "mean", "median", or "drop"
CATEGORICAL_ENCODING = "onehot"  # Options: "onehot", "label"

# Evaluation settings
TEST_SIZE = 0.2  # Proportion of data to use for testing
RANDOM_STATE = 42
