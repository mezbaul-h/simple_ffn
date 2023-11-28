import math
import random
import typing


def get_random_weight():
    return min(random.random() + 0.1, 0.99)


def make_random_matrix(row_size, column_size):
    return [
        [get_random_weight() for _ in range(column_size)]
        for _ in range(row_size)
    ]


def transpose_matrix(matrix):
    # Use nested list comprehensions to transpose the matrix
    transposed_matrix = [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    return transposed_matrix
