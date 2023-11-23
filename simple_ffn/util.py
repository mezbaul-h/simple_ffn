import math
import random
import typing


def _get_random_weight():
    return min(random.random() + 0.1, 0.99)
    # return 0.1


def generate_random_vector(dimension: typing.Union[int, typing.Tuple[int, int]]) -> typing.Union[typing.List[float], typing.List[typing.List[float]]]:
    """
    C or RxC
    """
    vector = []

    if isinstance(dimension, int):
        # 1D
        for _ in range(dimension):
            vector.append(_get_random_weight())
    else:
        # multi dimensional
        for i in range(dimension[0]):
            sub_vector = []

            for j in range(dimension[1]):
                sub_vector.append(_get_random_weight())

            vector.append(sub_vector)

    return vector


def calculate_rmses(calculated_outputs: typing.List[typing.List[float]], expected_outputs: typing.List[typing.List[float]]) -> typing.List[float]:
    squared_errors = []

    for i in range(len(calculated_outputs)):
        inner_calculated_outputs = calculated_outputs[i]
        inner_expected_outputs = expected_outputs[i]
        inner_squared_errors = []

        for j in range(len(inner_calculated_outputs)):
            inner_squared_errors.append((inner_expected_outputs[j] - inner_calculated_outputs[j])**2)

        squared_errors.append(inner_squared_errors)

    mean_squared_errors = {}
    n = 0

    for i in range(len(squared_errors)):
        for j in range(len(squared_errors[i])):
            if j not in mean_squared_errors:
                mean_squared_errors[j] = 0

            mean_squared_errors[j] += squared_errors[i][j]
            n += 1

    for k in mean_squared_errors:
        mean_squared_errors[k] = mean_squared_errors[k] / n

    return [math.sqrt(item) for item in mean_squared_errors.values()]


def scale_min_max(value, value_min, value_max):
    return (value - value_min) / (value_max - value_min)


def scale_matrix(matrix: typing.List[typing.List[float]], column_mins: typing.List[float] = None, column_maxes: typing.List[float] = None) -> typing.List[typing.List[float]]:
    if not matrix:
        return matrix

    num_rows = len(matrix)
    num_columns = len(matrix[0])

    if (column_mins is None) and (column_maxes is None):
        column_maxes = [float('-inf')] * num_columns
        column_mins = [float('inf')] * num_columns

        for i in range(num_rows):
            for j in range(num_columns):
                current_value = matrix[i][j]

                if column_maxes[j] < current_value:
                    column_maxes[j] = current_value

                if column_mins[j] > current_value:
                    column_mins[j] = current_value

    scaled_matrix = generate_random_vector((num_rows, num_columns))

    for i in range(num_rows):
        for j in range(num_columns):
            current_value = matrix[i][j]

            try:
                scaled_matrix[i][j] = scale_min_max(current_value, column_mins[j], column_maxes[j])
            except ZeroDivisionError:
                scaled_matrix[i][j] = 0.0

    return scaled_matrix
