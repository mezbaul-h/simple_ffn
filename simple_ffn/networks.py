import json
import time
import typing

from . import activations, layers

NeuronDataType = typing.Union[float, int]


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
        # self.layers[0].biases = [0 for _ in self.layers[0].biases]
        # self.layers[1].weights = [
        #     [0.33312217],
        #     [0.10090748],
        #     [0.47717155],
        #     [0.42678563],
        # ]
        # self.layers[1].biases = [0 for _ in self.layers[1].biases]

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

    def train(self, x, y, epochs: int = 100):
        """
        TODO: Randomize training batch each epoch.
        """
        for epoch in range(epochs):
            epoch_loss = 0

            for features, targets in zip(x, y):
                predictions = self.forward(features)
                losses = [prediction - target for prediction, target in zip(predictions, targets)]
                epoch_loss += 0.5 * (sum([loss**2 for loss in losses]) / len(losses))

                self.backward(losses)

            print(f'[{epoch + 1}/{epochs}] Training loss: {epoch_loss:.6f}')

            time.sleep(1 / 1000)

    def fit(self, x, y, epochs):
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

    def __call__(self, features):
        return self.predict(features)


    def get_score(self, x, y):
        return 1
