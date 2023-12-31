"""
Dataset Preprocessing Script

This script reads a dataset from a CSV file, performs preprocessing steps including scaling features and outputs, and
writes the preprocessed datasets to separate CSV files.
"""
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from simple_ffn.scalers import MinMaxScaler
from simple_ffn.utils import read_dataset_csv, train_test_split

DATASET_FILE_PREFIX = "ce889_dataCollection"
SOURCE_DATASET_PATH = PROJECT_ROOT / "data" / f"{DATASET_FILE_PREFIX}.csv"
TARGET_DIR = PROJECT_ROOT / "data"


def filter_duplicate_rows(data):
    """
    Filter duplicate rows based on the combination of values in columns 1 and 2, keeping the last occurrence.

    Parameters
    ----------
    data : list of lists
        The input data containing rows with columns 1 and 2.

    Returns
    -------
    list of lists
        Filtered data with only the last occurrence of each unique combination of values in columns 1 and 2.
    """
    # Create a dictionary to store unique rows with their last occurrence.
    unique_rows = {}

    for row in data:
        if not all([not item for item in row]):  # Skip rows with all zero values.
            key = tuple(row[:2])  # Use the first two elements as a key for comparison.
            unique_rows[key] = row

    # Convert the dictionary values back to a list of lists.
    result = list(unique_rows.values())

    return result


def write_dataset_csv(csv_filename, features, outputs):
    """
    Write a dataset to a CSV file.

    Parameters
    ----------
    csv_filename : Path
        The path to the CSV file.
    features : list
        The list of input features.
    outputs : list
        The list of output values.
    """
    with open(csv_filename, "w+") as csv_file:
        csv_writer = csv.writer(csv_file)

        for x, y in zip(features, outputs):
            csv_writer.writerow([*x, *y])


def main():
    """
    Main function to read, preprocess, and save datasets.
    """
    x, y = read_dataset_csv(SOURCE_DATASET_PATH, transformer=filter_duplicate_rows)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, random_state=42, test_size=0.1)

    # Scale features.
    feature_scaler = MinMaxScaler(x_train)
    x_train = feature_scaler.transform(x_train)

    with open(TARGET_DIR / "feature_scaler_params.json", "w+") as f:
        f.write(json.dumps(feature_scaler.get_params(), indent=4))

    # Scale outputs.
    output_scaler = MinMaxScaler(y_train)
    y_train = output_scaler.transform(y_train)

    with open(TARGET_DIR / "output_scaler_params.json", "w+") as f:
        f.write(json.dumps(output_scaler.get_params(), indent=4))

    # Write preprocessed datasets to CSV files.
    write_dataset_csv(TARGET_DIR / f"{DATASET_FILE_PREFIX}.train.csv", x_train, y_train)
    write_dataset_csv(TARGET_DIR / f"{DATASET_FILE_PREFIX}.validation.csv", x_validation, y_validation)
    write_dataset_csv(TARGET_DIR / f"{DATASET_FILE_PREFIX}.test.csv", x_test, y_test)


if __name__ == "__main__":
    main()
