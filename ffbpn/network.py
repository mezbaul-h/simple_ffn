import math
import random
import time
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

        for calculated_output, expected_output in zip(calculated_outputs, expected_outputs):
            loss_vector.append((expected_output - calculated_output)**2)
            loss_derivative_vector.append(-(expected_output - calculated_output))

        mse = sum(loss_vector) / len(loss_vector)
        rmse = math.sqrt(mse)

        return loss_vector, loss_derivative_vector, rmse

    def backward(self, losses, learning_rate):
        loss_vector, loss_derivative_vector, rmse = losses
        output_layer_index = len(self.layers) - 1
        gradients = []

        for current_layer_index in range(output_layer_index, -1, -1):
            current_layer = self.layers[current_layer_index]

            out_gradients = []
            sub_gradients = []

            if current_layer_index == output_layer_index:
                for i in range(current_layer.output_feature_count):
                    out_gradients.append(loss_derivative_vector[i] * current_layer.outputs[i] * (1 - current_layer.outputs[i]))

            for i in range(current_layer.input_feature_count):
                if out_gradients:
                    sub_gradient_item = sum([out_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)])
                else:
                    next_layer_gradients = gradients[0]
                    sub_gradient_item = sum([next_layer_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)])

                sub_gradients.append(sub_gradient_item)

            # calculate delta(w)s
            for i in range(current_layer.input_feature_count):
                for j in range(current_layer.output_feature_count):
                    if current_layer_index == output_layer_index:
                        delta_weight = out_gradients[j] * current_layer.inputs[i]
                    else:
                        next_layer_gradients = gradients[0]
                        delta_weight = sum([next_layer_gradients[j] * current_layer.weights[i][j] for j in range(current_layer.output_feature_count)]) * current_layer.inputs[i] * current_layer.outputs[j] * (1 - current_layer.outputs[j])

                    current_layer.delta_weights[i][j] = delta_weight

            # print(f'current layer {current_layer_index}')
            # print('cin', current_layer.inputs, 'cout', current_layer.outputs)
            # print('weights', current_layer.weights)
            # print('delta(w)', current_layer.delta_weights)
            # print('out gradients', out_gradients, 'sub gradients', sub_gradients)
            # print('-')

            gradients.insert(0, sub_gradients)

        # update weights
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def calculate_rmse(self, target_vector, calculated_vector):
        rmse_vector = [0.0] * len(target_vector)

        for target_inner_vector, calculated_inner_vector in zip(target_vector, calculated_vector):
            squared_diff = [(target - calculated)**2 for target, calculated in zip(target_inner_vector, calculated_inner_vector)]
            rmse_vector.append(math.sqrt(sum(squared_diff) / len(squared_diff)))

        return rmse_vector

    def fit(self, x, y, learning_rate=0.5, epochs=100):
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

            rmse = self.calculate_rmse([item[1] for item in combined_inputs_outputs], all_calculated_outputs)

            print(f'epoch: {i}, rmse: {sum(rmse) / len(rmse)}')

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
