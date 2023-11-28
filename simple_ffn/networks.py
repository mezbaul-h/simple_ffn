import json
import math
import time
import typing

from .activations import Sigmoid
from . import layers, activations
from .utils import transpose_matrix

NeuronDataType = typing.Union[float, int]


class Sequential:
    def __init__(self, *args: layers.Linear, **kwargs):
        self.learning_rate: float = kwargs.get('learning_rate')
        self.momentum: float = kwargs.get('momentum')

        self.layers = args
        self.num_layers = len(args)

        self._fix_layers()

        self.layers[0].weights = [
            [0.91300363, 0.94700325, 0.75501112, 0.76566442],
            [0.85213534, 0.88962154, 0.8392711,  0.60951791],
        ]
        self.layers[0].biases = [0 for _ in self.layers[0].biases]
        self.layers[1].weights = [
            [0.33312217],
            [0.10090748],
            [0.47717155],
            [0.42678563],
        ]
        self.layers[1].biases = [0 for _ in self.layers[1].biases]

    def _fix_layers(self):
        for i in range(self.num_layers):
            current_layer = self.layers[i]
            current_layer.next_layer = self.layers[i + 1] if i < (self.num_layers-1) else None
            current_layer.previous_layer = self.layers[i - 1] if i > 0 else None

            current_layer.learning_rate = self.learning_rate

            if current_layer.activation:
                current_layer.activation.learning_rate = self.learning_rate

    def forward(self, features):
        """
        Feeds forward a set of features.
        """
        for index, layer in enumerate(self.layers):
            features = layer(features)

        return features

    def backward(self, losses):
        carry_over_gradient = None

        for l_index in range(self.num_layers - 1, -1, -1):
            current_layer = self.layers[l_index]
            layer_gradients = []

            for i in range(current_layer.input_feature_count):
                for j in range(current_layer.output_feature_count):
                    if not current_layer.next_layer:  # output layer
                        layer_gradients.append(losses[0] * current_layer.weights[i][j] * current_layer.inputs[i] * (1 - current_layer.inputs[i]))

                        delta_bias = losses[j]
                        delta_weight = current_layer.inputs[i] * losses[j]
                    else:
                        delta_bias = carry_over_gradient[j]
                        delta_weight = current_layer.inputs[i] * carry_over_gradient[j]

                    current_layer.delta_weights[i][j] = delta_weight
                    current_layer.delta_biases[j] = delta_bias

            carry_over_gradient = layer_gradients

        for layer in self.layers:
            layer.update_biases_and_weights()

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

    def train(self, x, y, epochs: int = 100):
        """
        TODO: Randomize training batch each epoch.
        """
        self.validate_network(x, y, epochs)

        for epoch in range(epochs):
            epoch_loss = 0

            for features, targets in zip(x, y):
                predictions = self.forward(features)
                losses = [prediction - target for prediction, target in zip(predictions, targets)]
                epoch_loss += 0.5 * (sum([loss**2 for loss in losses]) / len(losses))

                self.backward(losses)

            print(f'[{epoch + 1}/{epochs}] Training loss: {epoch_loss:.6f}')

            time.sleep(1 / 1000)

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
            }, indent=4))

    def load(self, checkpoint_filename):
        with open(checkpoint_filename, 'r') as f:
            checkpoint_data = json.loads(f.read())

        activations = checkpoint_data['activations']
        layer_dimensions = checkpoint_data['layer_dimensions']
        self.layers = []

        for i in range(len(activations)):
            activation = activations[i]
            layer_dimension = layer_dimensions[i]

            self.layers.append(layers.Linear(input_feature_count=layer_dimension[0], output_feature_count=layer_dimension[1], activation=Sigmoid() if activation == 'sigmoid' else None))

        weight_matrices = checkpoint_data['weight_matrices']

        for i in range(len(weight_matrices)):
            self.layers[i].weights = weight_matrices[i]

        self.learning_rate = checkpoint_data['learning_rate']

    def __call__(self, features):
        predictions = self.forward(features)

        return predictions
