"""
model_lightgbm.py

This module defines the LGBMClassifier model for credit risk prediction.
It includes functions for training, saving, and loading the model.
"""

import joblib
from lightgbm import LGBMClassifier
import logging

def train_lightgbm_model(X_train, y_train) -> LGBMClassifier:
    """
    Trains an LGBMClassifier model on the provided training data.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training target variable.

    Returns:
        LGBMClassifier: The trained LightGBM model.
    """
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    logging.info("LightGBM model training complete.")
    return model

def save_lightgbm_model(model, file_path: str = "credit_risk_lightgbm_model.pkl"):
    """
    Saves the trained LightGBM model to disk using joblib.

    Args:
        model (LGBMClassifier): The trained LightGBM model.
        file_path (str): The path to save the model file.
    """
    joblib.dump(model, file_path)
    logging.info(f"LightGBM model saved to {file_path}")

def load_lightgbm_model(file_path: str = "credit_risk_lightgbm_model.pkl") -> LGBMClassifier:
    """
    Loads a trained LightGBM model from disk.

    Args:
        file_path (str): The path to the saved model file.

    Returns:
        LGBMClassifier: The loaded LightGBM model.
    """
    model = joblib.load(file_path)
    logging.info(f"LightGBM model loaded from {file_path}")
    return model

if __name__ == "__main__":
    import data_loader, preprocessing
    df = data_loader.load_data()
    df_clean = preprocessing.clean_data(df)
    if 'category' in df_clean.columns:
        df_clean = preprocessing.encode_categorical(df_clean, ['category'])
    X_train, X_test, y_train, y_test = preprocessing.split_data(df_clean, target_column='default')
    
    model = train_lightgbm_model(X_train, y_train)
    save_lightgbm_model(model)
