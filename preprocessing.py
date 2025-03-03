"""
preprocessing.py

This module includes functions for data cleaning and feature engineering
to prepare the dataset for model training and evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE, MISSING_VALUE_STRATEGY, CATEGORICAL_ENCODING

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values and performing basic cleaning tasks.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame with missing values handled.
    """
    if MISSING_VALUE_STRATEGY == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif MISSING_VALUE_STRATEGY == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif MISSING_VALUE_STRATEGY == "drop":
        df = df.dropna()
    else:
        raise ValueError("Invalid MISSING_VALUE_STRATEGY specified in config.")
    
    return df

def encode_categorical(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    Encodes categorical features using one-hot encoding or label encoding.

    Args:
        df (pd.DataFrame): The DataFrame containing categorical features.
        categorical_columns (list): A list of column names to be encoded.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical variables.
    """
    if CATEGORICAL_ENCODING == "onehot":
        df = pd.get_dummies(df, columns=categorical_columns)
    elif CATEGORICAL_ENCODING == "label":
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    else:
        raise ValueError("Invalid CATEGORICAL_ENCODING specified in config.")
    
    return df

def split_data(df: pd.DataFrame, target_column: str):
    """
    Splits the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        target_column (str): The column name of the target variable.

    Returns:
        tuple: A tuple containing training features, test features, training target, and test target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the preprocessing functions
    import data_loader
    df = data_loader.load_data()
    df_clean = clean_data(df)
    # Assume 'category' is a categorical column and 'default' is the target variable
    if 'category' in df_clean.columns:
        df_clean = encode_categorical(df_clean, ['category'])
    print(df_clean.head())
