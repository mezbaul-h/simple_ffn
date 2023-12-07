import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from simple_ffn.scalers import MinMaxScaler
from simple_ffn.utils import read_dataset_csv

DATASET_FILE_PREFIX = "ce889_dataCollection"
SOURCE_DATASET_PATH = PROJECT_ROOT / "lander" / f"{DATASET_FILE_PREFIX}.csv"
TARGET_DIR = PROJECT_ROOT / "data"


def train_test_split(features, outputs, test_size=0.2, random_state=None):
    """
    Split arrays or lists into random train and test subsets.

    Parameters
    ----------
    features : array-like or list
        The input data. Features to be split.
    outputs : array-like or list
        The labels or outputs associated with the input data.
    test_size : float, optional, default: 0.2
        Proportion of the dataset to include in the test split.
    random_state : int or None, optional, default: None
        Seed for random number generation. If None, a random seed will be used.

    Returns
    -------
    tuple
        A tuple containing four arrays or lists:
        - `features_train`: Features for training.
        - `features_test`: Features for testing.
        - `outputs_train`: Labels or outputs for training.
        - `outputs_test`: Labels or outputs for testing.

    Examples
    --------
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = [0, 1, 0, 1, 0, 1]
    >>> x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    """
    if random_state is not None:
        random.seed(random_state)

    data = list(zip(features, outputs))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))

    train_data = data[:split_index]
    test_data = data[split_index:]

    features_train, outputs_train = zip(*train_data)
    features_test, outputs_test = zip(*test_data)

    return list(features_train), list(features_test), list(outputs_train), list(outputs_test)


def write_dataset_csv(csv_filename, features, outputs):
    with open(csv_filename, "w+") as csv_file:
        csv_writer = csv.writer(csv_file)

        for x, y in zip(features, outputs):
            csv_writer.writerow([*x, *y])


def main():
    x, y = read_dataset_csv(SOURCE_DATASET_PATH, criterion=lambda row: all([float(column) != 0 for column in row]))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.1)

    feature_scaler = MinMaxScaler(x_train)
    x_train = feature_scaler.transform(x_train)

    with open(TARGET_DIR / "feature_scaler_params.json", "w+") as f:
        f.write(json.dumps(feature_scaler.get_params(), indent=4))

    output_scaler = MinMaxScaler(y_train)
    y_train = output_scaler.transform(y_train)

    with open(TARGET_DIR / "output_scaler_params.json", "w+") as f:
        f.write(json.dumps(output_scaler.get_params(), indent=4))

    write_dataset_csv(TARGET_DIR / f"{DATASET_FILE_PREFIX}.train.csv", x_train, y_train)
    write_dataset_csv(TARGET_DIR / f"{DATASET_FILE_PREFIX}.validation.csv", x_validation, y_validation)
    write_dataset_csv(TARGET_DIR / f"{DATASET_FILE_PREFIX}.test.csv", x_test, y_test)


if __name__ == "__main__":
    main()
