import math


class Sigmoid:
    def __init__(self):
        self.outputs = None
        self.inputs = None

    @staticmethod
    def calculate_sigmoid(item, learning_rate):
        # gamma = -learning_rate * item
        gamma = -item

        # To avoid math range error.
        # if gamma < 0:
        #     a = math.exp(gamma)
        #     activation_value = a / (1 + a)
        # else:
        #     activation_value = 1 / (1 + math.exp(-gamma))

        return 1 / (1 + math.exp(gamma))

    def __call__(self, x, learning_rate: float):
        self.inputs = x
        self.outputs = []

        for item in x:
            self.outputs.append(self.calculate_sigmoid(item, learning_rate))

        return self.outputs.copy()
