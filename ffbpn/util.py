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
