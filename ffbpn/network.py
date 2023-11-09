import typing

from .activation import Sigmoid
from .layer import Layer


class Network:
    def __init__(self, *args: Layer):
        self.layers = args

    def forward(self, x, learning_rate):
        """
        Feeds forward a set of inputs.
        """
        for layer in self.layers:
            x = layer(x, learning_rate)

        return x

    def loss(self, calculated_outputs, expected_outputs):
        """
        mse s
        """
        loss_vector = []
        loss_derivative_vector = []
        output_neuron_count = len(expected_outputs)

        for calculated_output, expected_output in zip(calculated_outputs, expected_outputs):
            # formula: 1/n * (expected - predicted)^2
            loss_vector.append((1/output_neuron_count) * (expected_output - calculated_output)**2)
            loss_derivative_vector.append(-(expected_output - calculated_output))

        return loss_vector, loss_derivative_vector

    def backward(self, losses):
        loss_vector, loss_derivative_vector = losses
        output_layer_index = len(self.layers) - 1
        gradients = []
        layer_index_cursor = output_layer_index
        derivative_matrix = []

        for current_layer_index in range(output_layer_index, -1, -1):
            current_layer = self.layers[current_layer_index]

            sub_derivative_matrix = []

            weight_matrix = current_layer.weights

            for row in weight_matrix:
                ssub_derivative_matrix = []
                for column in row:
                    ...

        print('-')

        return gradients

    def fit(self, x, y, learning_rate=0.5, epochs=1000):
        for input_item, expected_outputs in zip(x, y):
            calculated_outputs = self.forward(input_item, learning_rate)
            losses = self.loss(calculated_outputs, expected_outputs)
            self.backward(losses)
