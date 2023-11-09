import math


class Sigmoid:
    def __init__(self):
        self.outputs = None
        self.inputs = None

    def __call__(self, x, learning_rate: float):
        self.inputs = x
        self.outputs = []

        for item in x:
            self.outputs.append(1/(1 + math.exp(-learning_rate * item)))

        return self.outputs
