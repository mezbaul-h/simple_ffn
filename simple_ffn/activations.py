import math


class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.learning_rate = None
        self.outputs = None

    @staticmethod
    def calculate_sigmoid(value):
        gamma = -value

        return 1 / (1 + math.exp(gamma))

    @staticmethod
    def calculate_sigmoid_derivative(value):
        return value * (1 - value)

    def __call__(self, x):
        self.inputs = x
        self.outputs = []

        for item in x:
            self.outputs.append(self.calculate_sigmoid(item))

        return self.outputs.copy()
