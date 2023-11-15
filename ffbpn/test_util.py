import math

from .util import calculate_rmses


def test_rmse():
    calculated_output_vector = [[0, 0], [4, 4]]
    expected_output_vector = [[4, 4], [0, 0]]

    calculated_rmses = calculate_rmses(expected_output_vector, calculated_output_vector)
    assert calculated_rmses == [math.sqrt(8), math.sqrt(8)]
