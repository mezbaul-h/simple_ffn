import csv
import math
import random


def _get_random_weight():
    return min(random.random() + 0.1, 0.99)


def _get_zero_weight():
    return 0.0


def _make_matrix(row_size, column_size, value_func):
    return [[value_func() for _ in range(column_size)] for _ in range(row_size)]


def initialize_xavier_weights(input_size, output_size, random_state=None):
    """
    Initialize weights using the Xavier/Glorot Initialization.

    Parameters
    ----------
    input_size : int
        Number of input units.
    output_size : int
        Number of output units.
    random_state : int or None, optional
        Seed for reproducibility. If specified, the random number generator
        will be seeded for consistent results. Default is None.

    Returns
    -------
    weights : list
        Initialized weights with shape (input_size, output_size).

    Notes
    -----
    Xavier/Glorot Initialization scales the weights based on the number of
    input and output units to prevent vanishing/exploding gradients during
    training.

    The formula for standard deviation (std_dev) is:
        std_dev = sqrt(2 / (input_size + output_size))

    Examples
    --------
    >>> a = 256
    >>> b = 128
    >>> w = initialize_xavier_weights(a, b, random_state=42)
    """
    if random_state is not None:
        random.seed(random_state)

    variance = 2.0 / (input_size + output_size)
    std_dev = math.sqrt(variance)

    weights = [[random.gauss(0, std_dev) for _ in range(output_size)] for _ in range(input_size)]

    return weights


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
