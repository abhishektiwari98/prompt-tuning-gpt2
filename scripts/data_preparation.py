# scripts/data_preparation.py

import pandas as pd
import os


def prepare_custom_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    custom_df = pd.read_csv(file_path)
    if not all(column in custom_df.columns for column in ['text', 'category']):
        raise ValueError("Dataset must contain 'text' and 'category' columns.")
    print("Data preparation complete. Here's a preview of the dataset:")
    print(custom_df.head())


if __name__ == "__main__":
    prepare_custom_data('data/custom_data.csv')
