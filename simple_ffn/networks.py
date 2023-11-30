import json
import time
from . import activations, layers
from .utils import get_random_vector_indexes


class Sequential:
    def __init__(self, *args: layers.Linear, **kwargs):
        self.learning_rate: float = kwargs.get('learning_rate')
        self.momentum: float = kwargs.get('momentum')

        self.layers = args
        self.num_layers = len(args)

        self._fix_layers()

        # self.layers[0].weights = [
        #     [0.91300363, 0.94700325, 0.75501112, 0.76566442],
        #     [0.85213534, 0.88962154, 0.8392711,  0.60951791],
        # ]
        # self.layers[1].weights = [
        #     [0.33312217],
        #     [0.10090748],
        #     [0.47717155],
        #     [0.42678563],
        # ]

    def _fix_layers(self):
        for i in range(self.num_layers):
            current_layer = self.layers[i]
            current_layer.next_layer = self.layers[i + 1] if i < (self.num_layers-1) else None
            current_layer.previous_layer = self.layers[i - 1] if i > 0 else None

            current_layer.learning_rate = self.learning_rate
            current_layer.momentum = self.momentum

            if current_layer.activation:
                current_layer.activation.learning_rate = self.learning_rate

    def forward(self, features):
        """
        Feeds forward a set of features.
        """
        for index, layer in enumerate(self.layers):
            features = layer.forward(features)

        return features

    def backward(self, losses):
        carry_over_gradient = None

        for l_index in range(self.num_layers - 1, -1, -1):
            current_layer = self.layers[l_index]
            layer_gradients = []

            for i in range(current_layer.input_feature_count):
                for j in range(current_layer.output_feature_count):
                    if not current_layer.next_layer:  # output layer
                        layer_gradients.append(losses[0] * current_layer.weights[i][j] * activations.Sigmoid.calculate_sigmoid_derivative(current_layer.inputs[i]))

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

    def train(self, x_train, y_train, epochs: int = 1, x_validation=None, y_validation=None):
        for epoch in range(epochs):
            epoch_loss = 0

            if epoch:
                training_index_order = get_random_vector_indexes(len(x_train))
            else:
                training_index_order = list(range(len(x_train)))

            for index in training_index_order:
                features = x_train[index]
                targets = y_train[index]

                predictions = self.forward(features)
                losses = [prediction - target for prediction, target in zip(predictions, targets)]
                epoch_loss += 0.5 * (sum([loss**2 for loss in losses]) / len(losses))

                self.backward(losses)

            validation_loss = 0

            if x_validation and y_validation:
                # TODO: make sure x and y validation sets are scaled
                validation_loss = self.get_score(x_validation, y_validation)

            print(f'[{epoch + 1}/{epochs}] Training loss: {epoch_loss:.6f} Validation loss: {validation_loss:.6f}')

            time.sleep(1 / 1000)

    def fit(self, x, y, epochs: int = 1):
        self.train(x, y, epochs)

    def save(self, checkpoint_filename='checkpoint.json'):
        layer_params = []

        for i in range(len(self.layers)):
            current_layer = self.layers[i]

            layer_params.append({
                'sigmoid_activation': current_layer.activation is not None,
                'layer_dimensions': (current_layer.input_feature_count, current_layer.output_feature_count),
                'weights': current_layer.weights,
                'delta_weights': current_layer.delta_weights,
                'biases': current_layer.biases,
                'delta_biases': current_layer.delta_biases,

            })

        with open(checkpoint_filename, 'w+') as f:
            f.write(json.dumps({
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'layer_params': layer_params,
            }, indent=4))

    @classmethod
    def load(cls, checkpoint_filename):
        with open(checkpoint_filename, 'r') as f:
            checkpoint_data = json.loads(f.read())

        network_layers = []

        for layer_param in checkpoint_data['layer_params']:
            activation_function = None

            if layer_param['sigmoid_activation']:
                activation_function = activations.Sigmoid()

            layer = layers.Linear(layer_param['layer_dimensions'][0], layer_param['layer_dimensions'][1], activation_function)

            layer.weights = layer_param['weights']
            layer.delta_weights = layer_param['delta_weights']
            layer.biases = layer_param['biases']
            layer.delta_biases = layer_param['delta_biases']

            network_layers.append(layer)

        instance = cls(
            *network_layers,
            learning_rate=checkpoint_data['learning_rate'],
            momentum=checkpoint_data['momentum'],
        )

        return instance

    def predict(self, features):
        predictions = self.forward(features)

        return predictions

    def get_score(self, x, y):
        cum_loss = 0

        for features, targets in zip(x, y):
            predictions = self.forward(features)
            losses = [prediction - target for prediction, target in zip(predictions, targets)]
            cum_loss += 0.5 * (sum([loss ** 2 for loss in losses]) / len(losses))

        return cum_loss
