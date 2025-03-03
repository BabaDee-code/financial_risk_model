"""
data_loader.py

This module is responsible for loading the dataset required for the credit risk model.
It provides functions to load data from various sources such as CSV files.
"""

import pandas as pd
from config import DATA_FILE_PATH

def load_data(file_path: str = DATA_FILE_PATH) -> pd.DataFrame:
    """
    Load the credit risk dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the data. Defaults to DATA_FILE_PATH.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file is not found at the specified path.
        pd.errors.ParserError: If there is an error parsing the CSV.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        raise e
    except pd.errors.ParserError as e:
        print("Error parsing the CSV file.")
        raise e

if __name__ == "__main__":
    # For testing purposes
    df = load_data()
    print(df.head())
