import math
import random
import typing


def generate_random_vector(dimension: typing.Union[int, typing.Tuple[int, int]]) -> typing.Union[typing.List[float], typing.List[typing.List[float]]]:
    """
    C or RxC
    """
    vector = []

    if isinstance(dimension, int):
        # 1D
        for _ in range(dimension):
            vector.append(random.random())
    else:
        # multi dimensional
        for i in range(dimension[0]):
            sub_vector = []

            for j in range(dimension[1]):
                sub_vector.append(random.random())

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
