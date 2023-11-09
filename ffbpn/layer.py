from ffbpn.activation import Sigmoid
from ffbpn.util import generate_random_vector


class Layer:
    def __init__(self, input_feature_count: int, output_feature_count: int, activation: Sigmoid = None):
        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.inputs = None
        self.outputs = None
        self.weights = generate_random_vector((input_feature_count, output_feature_count))
        self.activation = activation

    def __call__(self, x, learning_rate):
        self.inputs = x
        self.outputs = [0] * self.output_feature_count

        for i in range(self.output_feature_count):
            self.outputs[i] = sum([x[j] * self.weights[j][i] for j in range(len(x))])

        if self.activation:
            self.outputs = self.activation(self.outputs, learning_rate)

        return self.outputs
