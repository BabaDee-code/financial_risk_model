"""
evaluation.py

This module provides functions to evaluate the performance of credit risk models.
It computes various evaluation metrics, including ROC curves and AUC, and plots them.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the trained model on the test set and calculates performance metrics.

    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): The test feature set.
        y_test (pd.Series): The true labels for the test set.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    print("Evaluation Metrics:", metrics)
    return metrics

def compute_auc(model, X_test, y_test) -> float:
    """
    Computes the AUC (Area Under the Curve) for the given model on the test set.

    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): The test feature set.
        y_test (pd.Series): The true labels for the test set.

    Returns:
        float: The AUC score.
    """
    # Use predict_proba if available; otherwise, use decision_function
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return auc(fpr, tpr)

def plot_roc_curve(model, X_test, y_test, model_name: str):
    """
    Plots the ROC curve for a single model.

    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): The test feature set.
        y_test (pd.Series): The true labels for the test set.
        model_name (str): The name of the model for labeling purposes.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_curves(models: dict, X_test, y_test, best_model_name: str = None):
    """
    Plots ROC curves for multiple models. Optionally, highlights the best performing model.

    Args:
        models (dict): A dictionary where keys are model names and values are trained model instances.
        X_test (pd.DataFrame): The test feature set.
        y_test (pd.Series): The true labels for the test set.
        best_model_name (str, optional): The name of the best performing model to highlight.
    """
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        lw = 3 if best_model_name and name == best_model_name else 1.5
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.show()
    
if __name__ == "__main__":
    # For testing purposes, you can integrate with your preprocessing and model modules.
    from model_rf import load_rf_model
    from model_catboost import load_catboost_model
    import data_loader, preprocessing
    df = data_loader.load_data()
    df_clean = preprocessing.clean_data(df)
    if 'category' in df_clean.columns:
        df_clean = preprocessing.encode_categorical(df_clean, ['category'])
    X_train, X_test, y_train, y_test = preprocessing.split_data(df_clean, target_column='default')
    
    rf_model = load_rf_model()
    catboost_model = load_catboost_model()
    
    print("Random Forest AUC:", compute_auc(rf_model, X_test, y_test))
    print("CatBoost AUC:", compute_auc(catboost_model, X_test, y_test))
    
    # Plot ROC curves for both models
    models = {"RandomForest": rf_model, "CatBoost": catboost_model}
    plot_roc_curves(models, X_test, y_test)
