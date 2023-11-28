import random


def _get_random_weight():
    return min(random.random() + 0.1, 0.99)


def _get_zero_weight():
    return 0.0


def _make_matrix(row_size, column_size, value_func):
    return [
        [value_func() for _ in range(column_size)]
        for _ in range(row_size)
    ]


def make_random_matrix(row_size, column_size):
    return _make_matrix(row_size, column_size, _get_random_weight)


def make_zeroes_matrix(row_size, column_size):
    return _make_matrix(row_size, column_size, _get_zero_weight)
