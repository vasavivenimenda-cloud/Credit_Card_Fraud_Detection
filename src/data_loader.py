import pandas as pd


def load_data(file_path):
    """
    Load the credit card dataset.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    try:
        data = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.")
        return data

    except FileNotFoundError:
        print("❌ Dataset not found.")
        return None