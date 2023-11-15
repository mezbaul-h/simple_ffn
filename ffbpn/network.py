import math
import random
import time
import typing

from .layer import Layer
from .util import calculate_rmses

NeuronDataType = typing.Union[float, int]


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

        for calculated_output, expected_output in zip(calculated_outputs, expected_outputs):
            loss_vector.append((expected_output - calculated_output)**2)
            loss_derivative_vector.append(-(expected_output - calculated_output))

        mse = sum(loss_vector) / len(loss_vector)
        rmse = math.sqrt(mse)

        return loss_vector, loss_derivative_vector, rmse

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
                sub_gradient_item = sum([out_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)])
            else:
                sub_gradient_item = sum([carry_over_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)])

            sub_gradients.append(sub_gradient_item)

        gradients.append(sub_gradients)

        return gradients

    @staticmethod
    def calculate_layer_delta_weights(current_layer, is_output_layer, carry_over_gradients, output_gradients):
        for i in range(current_layer.input_feature_count):
            for j in range(current_layer.output_feature_count):
                if is_output_layer:
                    delta_weight = output_gradients[j] * current_layer.inputs[i]
                else:
                    delta_weight = sum([carry_over_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)]) * current_layer.inputs[i] * current_layer.outputs[j] * (1 - current_layer.outputs[j])

                current_layer.delta_weights[i][j] = delta_weight

    def backward(self, losses, learning_rate):
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

            # print(f'current layer {current_layer_index}')
            # print('cin', current_layer.inputs, 'cout', current_layer.outputs)
            # print('weights', current_layer.weights)
            # print('delta(w)', current_layer.delta_weights)
            # print('out gradients', out_gradients, 'sub gradients', sub_gradients)
            # print('-')

        # update weights
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def validate_network(self, input_matrix: typing.List[typing.List[NeuronDataType]], output_matrix: typing.List[typing.List[NeuronDataType]], learning_rate: float, epochs: int):
        if not self.layers:
            raise AssertionError('no layers declared')

        if (learning_rate <= 0) or (learning_rate > 1):
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

    def fit(self, x: typing.List[typing.List[NeuronDataType]], y: typing.List[typing.List[NeuronDataType]], learning_rate: float = 0.5, epochs: int = 100):
        self.validate_network(x, y, learning_rate, epochs)

        context_switch_timer = 1 / 1000
        combined_inputs_outputs = list(zip(x, y))

        for i in range(epochs):
            all_calculated_outputs = []

            for input_vector, output_vector in combined_inputs_outputs:
                calculated_outputs = self.forward(input_vector, learning_rate)

                all_calculated_outputs.append(calculated_outputs)

                losses = self.loss(calculated_outputs, output_vector)

                self.backward(losses, learning_rate)

                time.sleep(context_switch_timer)

            rmses = calculate_rmses([item[1] for item in combined_inputs_outputs], all_calculated_outputs)

            print(f'epoch: {i}, rmse: {rmses}')

            random.shuffle(combined_inputs_outputs)

            time.sleep(context_switch_timer)

    def save(self):
        weight_matrix = []

        for i in range(len(self.layers)):
            weight_matrix.append(self.layers[i].weights)

        return weight_matrix

    def load(self, weight_matrix):
        for i in range(len(weight_matrix)):
            self.layers[i].weights = weight_matrix[i]

    def predict(self, x, learning_rate):
        calculated_outputs = self.forward(x, learning_rate)

        return calculated_outputs
