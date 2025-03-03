"""
model_xgboost.py

This module defines the XGBClassifier model for credit risk prediction.
It includes functions for training, saving, and loading the model.
"""

import joblib
from xgboost import XGBClassifier
import logging

def train_xgboost_model(X_train, y_train) -> XGBClassifier:
    """
    Trains an XGBClassifier model on the provided training data.

    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The training target variable.

    Returns:
        XGBClassifier: The trained XGBoost model.
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    logging.info("XGBoost model training complete.")
    return model

def save_xgboost_model(model, file_path: str = "credit_risk_xgboost_model.pkl"):
    """
    Saves the trained XGBoost model to disk using joblib.

    Args:
        model (XGBClassifier): The trained XGBoost model.
        file_path (str): The path to save the model file.
    """
    joblib.dump(model, file_path)
    logging.info(f"XGBoost model saved to {file_path}")

def load_xgboost_model(file_path: str = "credit_risk_xgboost_model.pkl") -> XGBClassifier:
    """
    Loads a trained XGBoost model from disk.

    Args:
        file_path (str): The path to the saved model file.

    Returns:
        XGBClassifier: The loaded XGBoost model.
    """
    model = joblib.load(file_path)
    logging.info(f"XGBoost model loaded from {file_path}")
    return model

if __name__ == "__main__":
    import data_loader, preprocessing
    df = data_loader.load_data()
    df_clean = preprocessing.clean_data(df)
    if 'category' in df_clean.columns:
        df_clean = preprocessing.encode_categorical(df_clean, ['category'])
    X_train, X_test, y_train, y_test = preprocessing.split_data(df_clean, target_column='default')
    
    model = train_xgboost_model(X_train, y_train)
    save_xgboost_model(model)
