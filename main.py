"""
main.py

The main script for the Financial and Credit Risk Model Development project.
This script orchestrates data loading, preprocessing, training of multiple models,
evaluation, and visualization of ROC curves to identify the best performing model.
"""

from data_loader import load_data
from preprocessing import clean_data, encode_categorical, split_data
from model_rf import train_rf_model, save_rf_model
from model_catboost import train_catboost_model, save_catboost_model
from model_xgboost import train_xgboost_model, save_xgboost_model
from model_lightgbm import train_lightgbm_model, save_lightgbm_model
from evaluation import evaluate_model, compute_auc, plot_roc_curves
from utils import setup_logging

def main():
    """
    Main function to run the full credit risk modeling pipeline.
    Trains multiple models, evaluates them, and plots ROC curves to highlight the best performing model.
    """
    setup_logging()
    
    # Load the dataset
    df = load_data()
    
    # Clean the data
    df_clean = clean_data(df)
    
    # Encode categorical variables if present
    categorical_columns = ['category'] if 'category' in df_clean.columns else []
    if categorical_columns:
        df_clean = encode_categorical(df_clean, categorical_columns)
    
    # Split the data (assume 'default' is the target variable)
    X_train, X_test, y_train, y_test = split_data(df_clean, target_column='default')
    
    # Train models
    rf_model = train_rf_model(X_train, y_train)
    catboost_model = train_catboost_model(X_train, y_train)
    xgboost_model = train_xgboost_model(X_train, y_train)
    lightgbm_model = train_lightgbm_model(X_train, y_train)
    
    # Optionally, save models
    save_rf_model(rf_model)
    save_catboost_model(catboost_model)
    save_xgboost_model(xgboost_model)
    save_lightgbm_model(lightgbm_model)
    
    # Evaluate all models
    models = {
        "RandomForest": rf_model,
        "CatBoost": catboost_model,
        "XGBoost": xgboost_model,
        "LightGBM": lightgbm_model
    }
    
    model_auc = {}
    for name, model in models.items():
        print(f"{name} Evaluation:")
        evaluate_model(model, X_test, y_test)
        auc_score = compute_auc(model, X_test, y_test)
        print(f"{name} AUC: {auc_score}\n")
        model_auc[name] = auc_score
    
    # Determine the best performing model based on AUC
    best_model_name = max(model_auc, key=model_auc.get)
    print(f"Best performing model based on AUC: {best_model_name}")
    
    # Plot ROC curves for all models, highlighting the best performing one
    plot_roc_curves(models, X_test, y_test, best_model_name=best_model_name)

if __name__ == "__main__":
    main()
