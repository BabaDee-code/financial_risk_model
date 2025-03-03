"""
model_rf.py

This module defines the RandomForestClassifier model for credit risk prediction.
It includes functions for training, saving, and loading the model.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from config import MODEL_PARAMS
import logging

def train_rf_model(X_train, y_train) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier model on the provided training data.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training target variable.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    logging.info("Random Forest model training complete.")
    return model

def save_rf_model(model, file_path: str = "credit_risk_rf_model.pkl"):
    """
    Saves the trained Random Forest model to disk using joblib.

    Args:
        model (RandomForestClassifier): The trained model.
        file_path (str): The path to save the model file.
    """
    joblib.dump(model, file_path)
    logging.info(f"Random Forest model saved to {file_path}")

def load_rf_model(file_path: str = "credit_risk_rf_model.pkl") -> RandomForestClassifier:
    """
    Loads a trained Random Forest model from disk.

    Args:
        file_path (str): The path to the saved model file.

    Returns:
        RandomForestClassifier: The loaded model.
    """
    model = joblib.load(file_path)
    logging.info(f"Random Forest model loaded from {file_path}")
    return model

if __name__ == "__main__":
    import data_loader, preprocessing
    df = data_loader.load_data()
    df_clean = preprocessing.clean_data(df)
    if 'category' in df_clean.columns:
        df_clean = preprocessing.encode_categorical(df_clean, ['category'])
    X_train, X_test, y_train, y_test = preprocessing.split_data(df_clean, target_column='default')
    
    model = train_rf_model(X_train, y_train)
    save_rf_model(model)
