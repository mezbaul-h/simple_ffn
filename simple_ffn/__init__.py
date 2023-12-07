from . import activations, layers, networks
from .datasets import Dataset


def main():
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 200

    dataset = Dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = dataset.process()

    network = networks.Sequential(
        layers.Linear(2, 2, activation=activations.Sigmoid()),
        layers.Linear(2, 2),
        feature_scaler=dataset.feature_scaler,
        output_scaler=dataset.output_scaler,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
    )

    try:
        network.train(x_train, y_train, x_validation, y_validation)

        # for x, y in zip(x_train, y_train):
        #     print(x, y, network.predict(x))
    except KeyboardInterrupt:
        ...

    network.save("ffn_checkpoint.json")
