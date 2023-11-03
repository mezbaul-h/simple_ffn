import math


class Sigmoid:
    def __init__(self):
        ...

    def __call__(self, x, learning_rate: float):
        outputs = []

        for item in x:
            outputs.append(1/(1 + math.exp(-learning_rate * item)))

        return outputs
