import math


class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.learning_rate = None
        self.outputs = None

    @staticmethod
    def calculate_sigmoid(value):
        # gamma = -learning_rate * item
        gamma = -value

        # To avoid math range error.
        # if gamma < 0:
        #     a = math.exp(gamma)
        #     activation_value = a / (1 + a)
        # else:
        #     activation_value = 1 / (1 + math.exp(-gamma))

        return 1 / (1 + math.exp(gamma))

    @staticmethod
    def calculate_sigmoid_derivative(value):
        return Sigmoid.calculate_sigmoid(value) * (1 - Sigmoid.calculate_sigmoid(value))

    def __call__(self, x):
        self.inputs = x
        self.outputs = []

        for item in x:
            self.outputs.append(self.calculate_sigmoid(item))

        return self.outputs.copy()
