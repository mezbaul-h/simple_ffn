from pathlib import Path

from . import activations, layers, networks
from .datasets import Dataset
from .settings import DEFAULT_CHECKPOINT_FILENAME


def main():
    dataset = Dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = dataset.process()

    if Path(DEFAULT_CHECKPOINT_FILENAME).is_file():
        network = networks.Sequential.load(DEFAULT_CHECKPOINT_FILENAME)
    else:
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 1

        network = networks.Sequential(
            layers.Linear(2, 2, activation=activations.Sigmoid(), random_state=42),
            layers.Linear(2, 2, random_state=43),
            feature_scaler=dataset.feature_scaler,
            learning_rate=learning_rate,
            momentum=momentum,
            num_epochs=num_epochs,
            output_scaler=dataset.output_scaler,
        )

    try:
        network.train(x_train, y_train, x_validation, y_validation)

        # for x, y in zip(x_train, y_train):
        #     print(x, y, network.predict(x))
    except KeyboardInterrupt:
        ...

    network.save("ffn_checkpoint.json")
    network.save_loss_plot()
