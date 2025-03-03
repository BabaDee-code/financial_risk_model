"""
model_catboost.py

This module defines the CatBoostClassifier model for credit risk prediction.
It includes functions for training, saving, and loading the CatBoost model.
"""

import joblib
from catboost import CatBoostClassifier
import logging

def train_catboost_model(X_train, y_train) -> CatBoostClassifier:
    """
    Trains a CatBoostClassifier model on the provided training data.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training target variable.

    Returns:
        CatBoostClassifier: The trained CatBoost model.
    """
    # Using some default hyperparameters; adjust as needed
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        random_seed=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    logging.info("CatBoost model training complete.")
    return model

def save_catboost_model(model, file_path: str = "credit_risk_catboost_model.pkl"):
    """
    Saves the trained CatBoost model to disk using joblib.

    Args:
        model (CatBoostClassifier): The trained CatBoost model.
        file_path (str): The path to save the model file.
    """
    joblib.dump(model, file_path)
    logging.info(f"CatBoost model saved to {file_path}")

def load_catboost_model(file_path: str = "credit_risk_catboost_model.pkl") -> CatBoostClassifier:
    """
    Loads a trained CatBoost model from disk.

    Args:
        file_path (str): The path to the saved model file.

    Returns:
        CatBoostClassifier: The loaded CatBoost model.
    """
    model = joblib.load(file_path)
    logging.info(f"CatBoost model loaded from {file_path}")
    return model

if __name__ == "__main__":
    import data_loader, preprocessing
    df = data_loader.load_data()
    df_clean = preprocessing.clean_data(df)
    if 'category' in df_clean.columns:
        df_clean = preprocessing.encode_categorical(df_clean, ['category'])
    X_train, X_test, y_train, y_test = preprocessing.split_data(df_clean, target_column='default')
    
    model = train_catboost_model(X_train, y_train)
    save_catboost_model(model)
