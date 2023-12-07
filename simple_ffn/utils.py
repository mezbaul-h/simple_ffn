import csv
import random


def _get_random_weight():
    return min(random.random() + 0.1, 0.99)


def _get_zero_weight():
    return 0.0


def _make_matrix(row_size, column_size, value_func):
    return [[value_func() for _ in range(column_size)] for _ in range(row_size)]


def make_random_matrix(row_size, column_size):
    return _make_matrix(row_size, column_size, _get_random_weight)


def make_zeroes_matrix(row_size, column_size):
    return _make_matrix(row_size, column_size, _get_zero_weight)


def get_random_vector_indexes(vector_size):
    random_indexes = []
    taken = {}

    for _ in range(vector_size):
        while True:
            random_index = random.randint(0, vector_size - 1)

            if random_index not in taken:
                taken[random_index] = True
                random_indexes.append(random_index)
                break

    return random_indexes


def read_dataset_csv(csv_filename, criterion=None):
    """
    Read a CSV file containing a dataset and extract features and outputs
    based on the provided criterion.

    Parameters
    ----------
    csv_filename : str or pathlib.Path
        The path to the CSV file. It can be either a string or a Path object.
    criterion : callable, optional
        A function used to filter rows. If provided, only rows satisfying the
        criterion will be included in the output. If not provided, all rows
        are included.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - List of features, where each feature is represented as a list of floats.
        - List of outputs, where each output is represented as a list of floats.

    Usage
    -----
    >>> SOURCE_DATASET_PATH = "path/to/your/dataset.csv"
    >>> f = lambda r: all([float(c) > 0 for c in r])
    >>> x, y = read_dataset_csv(SOURCE_DATASET_PATH, criterion)
    """
    features = []
    outputs = []

    with open(csv_filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            if row and ((not criterion) or criterion(row)):
                features.append([float(item) for item in row[:2]])
                outputs.append([float(item) for item in row[2:]])

    return features, outputs
