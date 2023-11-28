import typing

from simple_ffn.activations import Sigmoid
from simple_ffn.utils import make_random_matrix, make_zeroes_matrix


class Linear:
    def __init__(self, input_feature_count: int, output_feature_count: int, activation: Sigmoid = None):
        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.inputs = None
        self.learning_rate = None
        self.momentum = None
        self.outputs = None

        self.biases = make_zeroes_matrix(1, output_feature_count)[0]
        self.biases_momentum = make_zeroes_matrix(1, output_feature_count)[0]
        self.delta_biases = make_zeroes_matrix(1, output_feature_count)[0]

        self.weights = make_random_matrix(input_feature_count, output_feature_count)
        self.weights_momentum = make_zeroes_matrix(input_feature_count, output_feature_count)
        self.delta_weights = make_zeroes_matrix(input_feature_count, output_feature_count)

        self.activation = activation
        self.next_layer: typing.Optional[Linear] = None
        self.previous_layer: typing.Optional[Linear] = None

    def _calculate_momenta(self):
        for i in range(self.output_feature_count):
            self.biases_momentum[i] = (self.momentum * self.biases_momentum[i]) - (self.learning_rate * self.delta_biases[i])

        for i in range(self.input_feature_count):
            for j in range(self.output_feature_count):
                self.weights_momentum[i][j] = (self.momentum * self.weights_momentum[i][j]) - (self.learning_rate * self.delta_weights[i][j])

    def update_biases_and_weights(self):
        self._calculate_momenta()

        for i in range(self.output_feature_count):
            self.biases[i] += self.biases_momentum[i]

        for i in range(self.input_feature_count):
            for j in range(self.output_feature_count):
                self.weights[i][j] += self.weights_momentum[i][j]

    def forward(self, features):
        """
        Does forward propagation.
        """
        self.inputs = features
        self.outputs = [0] * self.output_feature_count

        for i in range(self.output_feature_count):
            self.outputs[i] = sum([features[j] * self.weights[j][i] for j in range(len(features))]) + self.biases[i]

        if self.activation:
            self.outputs = self.activation(self.outputs)

        return self.outputs.copy()
