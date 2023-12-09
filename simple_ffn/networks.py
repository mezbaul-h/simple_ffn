import json
import time

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from . import activations, layers
from .scalers import MinMaxScaler
from .utils import get_random_vector_indexes


class Sequential:
    def __init__(self, *args: layers.Linear, **kwargs):
        self.learning_rate: float = kwargs.get("learning_rate")
        self.momentum: float = kwargs.get("momentum")

        self.layers = args
        self.num_layers = len(args)

        self._fix_layers()

        self.feature_scaler = kwargs.get("feature_scaler")
        self.output_scaler = kwargs.get("output_scaler")

        self.num_epochs = kwargs["num_epochs"]
        self.current_epoch = 0

        self.epoch_losses = {
            "training": [],
            "validation": [],
        }
        self.layer_params_at_lowest_training_epoch = [layer.get_params() for layer in self.layers]
        self.layer_params_at_lowest_validation_epoch = [layer.get_params() for layer in self.layers]

        # self.layers[0].weights = [
        #     [0.91300363, 0.94700325, 0.75501112, 0.76566442],
        #     [0.85213534, 0.88962154, 0.8392711,  0.60951791],
        # ]
        # self.layers[1].weights = [
        #     [0.91300363, 0.94700325],
        #     [0.75501112, 0.76566442],
        #     [0.85213534, 0.88962154],
        #     [0.8392711, 0.60951791],
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
            current_layer.next_layer = self.layers[i + 1] if i < (self.num_layers - 1) else None
            current_layer.previous_layer = self.layers[i - 1] if i > 0 else None

            current_layer.learning_rate = self.learning_rate
            current_layer.momentum = self.momentum

            if current_layer.activation:
                current_layer.activation.learning_rate = self.learning_rate

    def evaluate(self, x_test, y_test, use_best_layer_params=False):
        old_layer_params = None

        # If using best layer params, record current layer states and load the
        # best states.
        if use_best_layer_params and self.layer_params_at_lowest_validation_epoch:
            old_layer_params = []

            for i, layer in enumerate(self.layers):
                old_layer_params.append(layer.get_params())
                layer.load_params(self.layer_params_at_lowest_validation_epoch[i])

        total_evaluation_losses = [0.0] * len(y_test[0])
        avg_evaluation_losses = total_evaluation_losses.copy()
        num_samples = len(x_test)

        for features, target_outputs in zip(x_test, y_test):
            # Validation features are not scaled, so that needs to go through
            # same transformation.
            scaled_features = self.feature_scaler.transform([features])[0]

            calculated_outputs = self.predict(scaled_features)

            # Target outputs are not scaled as well.
            scaled_target_outputs = self.output_scaler.transform([target_outputs])[0]

            losses = [prediction - target for prediction, target in zip(calculated_outputs, scaled_target_outputs)]

            for i in range(len(losses)):
                total_evaluation_losses[i] += losses[i] ** 2

        for i in range(len(total_evaluation_losses)):
            avg_evaluation_losses[i] = (total_evaluation_losses[i] / num_samples) ** 0.5

        # If using best layer params, reload old layer states.
        if old_layer_params:
            for i, layer in enumerate(self.layers):
                layer.load_params(old_layer_params[i])

        return avg_evaluation_losses

    def save_loss_plot(self, target_filename="loss_plot.png"):
        # Don't try to plot if library is not available.
        if not plt:
            return

        training_losses = self.epoch_losses["training"]
        best_training_epoch_index = self.get_best_epoch(training_losses, self.current_epoch)
        validation_losses = self.epoch_losses["validation"]
        best_validation_epoch_index = self.get_best_epoch(validation_losses, self.current_epoch)

        # Create a line plot
        plt.figure(figsize=(10, 6))  # 10 * 100 x 6 * 100 pixels
        plt.plot(range(1, len(training_losses) + 1), training_losses, color="red", label="Training Loss")
        plt.axvline(
            x=best_training_epoch_index + 1, color="red", label=f"Best Training Epoch: {best_training_epoch_index + 1}"
        )

        plt.plot(range(1, len(validation_losses) + 1), validation_losses, color="green", label="Validation Loss")
        plt.axvline(
            x=best_validation_epoch_index + 1,
            color="green",
            label=f"Best Validation Epoch: {best_validation_epoch_index + 1}",
        )

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "Training and Validation Loss Over Epochs\n"
            f"Learning Rate: {self.learning_rate}, Momentum: {self.momentum}"
        )
        # Show the legend
        plt.legend()

        # Save the plot to a PNG file
        plt.savefig(target_filename)

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
                if current_layer.previous_layer:
                    layer_gradients.append(
                        sum(
                            [
                                losses[k] * current_layer.weights[i][k]
                                for k in range(current_layer.output_feature_count)
                            ]
                        )
                        * current_layer.previous_layer.activation.derivative(current_layer.inputs[i])
                    )

                for j in range(current_layer.output_feature_count):
                    if not current_layer.next_layer:  # output layer
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

    @staticmethod
    def get_best_epoch(epoch_losses, default):
        try:
            return epoch_losses.index(min(epoch_losses))
        except ValueError:
            return default

    def train(self, x_train, y_train, x_validation, y_validation, log_prefix=None):
        num_samples = len(x_train)

        while self.current_epoch < self.num_epochs:
            epoch_start_time = time.time()

            total_training_losses = [0.0] * len(y_train[0])
            avg_training_losses = total_training_losses.copy()

            if self.current_epoch:
                training_index_order = get_random_vector_indexes(len(x_train))
            else:
                training_index_order = list(range(len(x_train)))

            for index in training_index_order:
                features = x_train[index]
                targets = y_train[index]

                predictions = self.forward(features)

                losses = [prediction - target for prediction, target in zip(predictions, targets)]

                for i in range(len(losses)):
                    total_training_losses[i] += losses[i] ** 2

                self.backward(losses)

            for i in range(len(avg_training_losses)):
                avg_training_losses[i] = (total_training_losses[i] / num_samples) ** 0.5

            avg_validation_losses = self.evaluate(x_validation, y_validation)

            mean_training_loss = sum(avg_training_losses) / len(avg_training_losses)
            mean_validation_loss = sum(avg_validation_losses) / len(avg_validation_losses)

            self.epoch_losses["training"].append(mean_training_loss)
            self.epoch_losses["validation"].append(mean_validation_loss)

            best_training_epoch_index = self.get_best_epoch(self.epoch_losses["training"], self.current_epoch)

            # Save layer parameters (weights, biases) if current epoch is best epoch.
            if best_training_epoch_index == self.current_epoch:
                self.layer_params_at_lowest_training_epoch = [layer.get_params() for layer in self.layers]

            best_validation_epoch_index = self.get_best_epoch(self.epoch_losses["validation"], self.current_epoch)

            if best_validation_epoch_index == self.current_epoch:
                self.layer_params_at_lowest_validation_epoch = [layer.get_params() for layer in self.layers]

            seconds_elapsed = time.time() - epoch_start_time

            print(
                f"{log_prefix or ''}"
                f"[{self.current_epoch + 1}/{self.num_epochs}] "
                f"Losses (T/V): {mean_training_loss:.10f}/{mean_validation_loss:.10f} | "
                f"Best Epochs (T/V): {best_training_epoch_index + 1}/{best_validation_epoch_index + 1} "
                f"({seconds_elapsed:.2f}s)"
            )

            self.current_epoch += 1

    def get_params(self):
        return {
            "current_epoch": self.current_epoch,
            "epoch_losses": self.epoch_losses,
            "feature_scaler_params": self.feature_scaler.get_params(),
            "layer_params": [layer.get_params() for layer in self.layers],
            "layer_params_at_lowest_training_epoch": self.layer_params_at_lowest_training_epoch,
            "layer_params_at_lowest_validation_epoch": self.layer_params_at_lowest_validation_epoch,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "num_epochs": self.num_epochs,
            "output_scaler_params": self.output_scaler.get_params(),
        }

    def save(self, checkpoint_filename="checkpoint.json"):
        with open(checkpoint_filename, "w+") as f:
            f.write(json.dumps(self.get_params(), indent=4))

    @classmethod
    def load(cls, checkpoint_filename):
        with open(checkpoint_filename, "r") as f:
            checkpoint_data = json.loads(f.read())

        network_layers = []

        for layer_param in checkpoint_data["layer_params"]:
            activation_function = None

            if layer_param["sigmoid_activation"]:
                activation_function = activations.Sigmoid()

            layer = layers.Linear(
                layer_param["layer_dimensions"][0], layer_param["layer_dimensions"][1], activation_function
            )

            layer.weights = layer_param["weights"]
            layer.delta_weights = layer_param["delta_weights"]
            layer.biases = layer_param["biases"]
            layer.delta_biases = layer_param["delta_biases"]

            network_layers.append(layer)

        feature_scaler = MinMaxScaler()
        feature_scaler.load_params(checkpoint_data["feature_scaler_params"])

        output_scaler = MinMaxScaler()
        output_scaler.load_params(checkpoint_data["output_scaler_params"])

        instance = cls(
            *network_layers,
            feature_scaler=feature_scaler,
            learning_rate=checkpoint_data["learning_rate"],
            momentum=checkpoint_data["momentum"],
            num_epochs=checkpoint_data["num_epochs"],
            output_scaler=output_scaler,
        )

        instance.current_epoch = checkpoint_data["current_epoch"]
        instance.epoch_losses = checkpoint_data["epoch_losses"]
        instance.layer_params_at_lowest_training_epoch = checkpoint_data["layer_params_at_lowest_training_epoch"]
        instance.layer_params_at_lowest_validation_epoch = checkpoint_data["layer_params_at_lowest_validation_epoch"]

        return instance

    def predict(self, features):
        predictions = self.forward(features)

        return predictions
