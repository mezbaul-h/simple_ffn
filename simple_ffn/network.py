import json
import math
import random
import time
import typing

from .activation import Sigmoid
from .layer import Layer
from .util import calculate_rmses

NeuronDataType = typing.Union[float, int]


class Network:
    def __init__(self, *args: Layer, **kwargs):
        self.layers = args
        self.learning_rate = kwargs.get('learning_rate')

    def forward(self, x):
        """
        Feeds forward a set of inputs.
        """
        for layer in self.layers:
            x = layer(x, self.learning_rate)

        return x

    def loss(self, calculated_outputs, expected_outputs):
        loss_vector = []
        loss_derivative_vector = []

        for calculated_output, expected_output in zip(calculated_outputs, expected_outputs):
            loss_vector.append((expected_output - calculated_output)**2)
            loss_derivative_vector.append(expected_output - calculated_output)

        mse = sum(loss_vector) / len(loss_vector)
        rmse = math.sqrt(mse)

        return loss_vector, loss_derivative_vector, mse, rmse

    @staticmethod
    def calculate_layer_gradients(current_layer, is_output_layer, carry_over_gradients, losses):
        loss_derivative_vector = losses[1]
        gradients = []
        out_gradients = []
        sub_gradients = []

        if is_output_layer:
            for i in range(current_layer.output_feature_count):
                out_gradients.append(loss_derivative_vector[i] * current_layer.outputs[i] * (1 - current_layer.outputs[i]))

            gradients.append(out_gradients)

        for i in range(current_layer.input_feature_count):
            if out_gradients:
                sub_gradient_item = current_layer.inputs[i] * (1 - current_layer.inputs[i]) * sum([out_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)])
            else:
                sub_gradient_item = current_layer.inputs[i] * (1 - current_layer.inputs[i]) * sum([carry_over_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)])

            sub_gradients.append(sub_gradient_item)

        gradients.append(sub_gradients)

        return gradients

    def calculate_layer_delta_weights(self, current_layer, is_output_layer, carry_over_gradients, output_gradients):
        for i in range(current_layer.input_feature_count):
            for j in range(current_layer.output_feature_count):
                if is_output_layer:
                    delta_weight = self.learning_rate * output_gradients[j] * current_layer.inputs[i]
                else:
                    # delta_weight = self.learning_rate * sum([carry_over_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)]) * current_layer.inputs[i]
                    delta_weight = self.learning_rate * carry_over_gradients[j] * current_layer.inputs[i]

                current_layer.delta_weights[i][j] = delta_weight

    def backward(self, losses):
        output_layer_index = len(self.layers) - 1
        carry_over_gradients = None

        for current_layer_index in range(output_layer_index, -1, -1):
            current_layer = self.layers[current_layer_index]
            is_output_layer = current_layer_index == output_layer_index

            # calculate gradients
            gradients = self.calculate_layer_gradients(current_layer, is_output_layer, carry_over_gradients, losses)

            output_gradients = gradients[0] if is_output_layer else None

            # calculate delta(w)s
            self.calculate_layer_delta_weights(current_layer, is_output_layer, carry_over_gradients, output_gradients)

            carry_over_gradients = gradients.pop()

        # update weights
        for layer in self.layers:
            layer.update_weights(self.learning_rate)

    def validate_network(self, input_matrix: typing.List[typing.List[NeuronDataType]], output_matrix: typing.List[typing.List[NeuronDataType]], epochs: int):
        if not self.layers:
            raise AssertionError('no layers declared')

        if (self.learning_rate <= 0) or (self.learning_rate > 1):
            raise ValueError(f'learning rate must be in the range of (0, 1]')

        if epochs <= 0:
            raise ValueError('epoch count cannot be a negative number or zero')

        input_row_count = len(input_matrix)
        output_row_count = len(output_matrix)

        if input_row_count != output_row_count:
            raise AssertionError(f'{input_row_count} input rows provided against {output_row_count} output rows')

        if not input_row_count:
            raise AssertionError('no input data provided')

        input_feature_count = len(input_matrix[0])
        output_feature_count = len(output_matrix[0])

        input_layer = self.layers[0]
        output_layer = self.layers[-1]

        if input_layer.input_feature_count != input_feature_count:
            raise AssertionError(f'networks input layer is declared to have {input_layer.input_feature_count} features but {input_feature_count} were provided')

        if output_layer.output_feature_count != output_feature_count:
            raise AssertionError(f'networks output layer is declared to have {output_layer.output_feature_count} features but {output_feature_count} were provided')

    def fit(self, x: typing.List[typing.List[NeuronDataType]], y: typing.List[typing.List[NeuronDataType]], epochs: int = 100):
        self.validate_network(x, y, epochs)

        context_switch_timer = 1 / 1000
        input_row_count = len(x)
        combined_inputs_outputs = list(zip(x, y))

        for epoch in range(epochs):
            total_mse = 0

            for input_vector, output_vector in combined_inputs_outputs:
                calculated_outputs = self.forward(input_vector)

                losses = self.loss(calculated_outputs, output_vector)

                total_mse += losses[2]

                self.backward(losses)

            rmse = math.sqrt(total_mse / input_row_count)

            print(f'[{epoch + 1}/{epochs}] Training loss: {rmse:.4f}')

            random.shuffle(combined_inputs_outputs)

            time.sleep(context_switch_timer)

    def save(self, checkpoint_filename='checkpoint.json'):
        activations = []
        layer_dimensions = []
        weight_matrices = []

        for i in range(len(self.layers)):
            activations.append('sigmoid' if self.layers[i].activation else None)
            layer_dimensions.append((self.layers[i].input_feature_count, self.layers[i].output_feature_count))
            weight_matrices.append(self.layers[i].weights)

        with open(checkpoint_filename, 'w+') as f:
            f.write(json.dumps({
                'activations': activations,
                'layer_dimensions': layer_dimensions,
                'learning_rate': self.learning_rate,
                'weight_matrices': weight_matrices,
            }))

    def load(self, checkpoint_filename):
        with open(checkpoint_filename, 'r') as f:
            checkpoint_data = json.loads(f.read())

        activations = checkpoint_data['activations']
        layer_dimensions = checkpoint_data['layer_dimensions']
        self.layers = []

        for i in range(len(activations)):
            activation = activations[i]
            layer_dimension = layer_dimensions[i]

            self.layers.append(Layer(input_feature_count=layer_dimension[0], output_feature_count=layer_dimension[1], activation=Sigmoid() if activation == 'sigmoid' else None))

        weight_matrices = checkpoint_data['weight_matrices']

        for i in range(len(weight_matrices)):
            self.layers[i].weights = weight_matrices[i]

        self.learning_rate = checkpoint_data['learning_rate']

    def predict(self, x):
        calculated_outputs = self.forward(x)

        return calculated_outputs
