import math


class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.learning_rate = None
        self.outputs = None

    @staticmethod
    def activate(value):
        try:
            return 1 / (1 + math.exp(-value))
        except OverflowError:
            return 1 - (1 / 1 + math.exp(value))

    @staticmethod
    def derivative(value):
        return value * (1 - value)

    def forward(self, x):
        self.inputs = x
        self.outputs = []

        for item in x:
            self.outputs.append(self.activate(item))

        return self.outputs
