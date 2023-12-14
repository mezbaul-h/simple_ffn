"""
Simple FFN Package

This package provides a simple feedforward neural network implementation.
"""
from pathlib import Path

from . import activations, layers, networks
from .arg_parsers import make_main_arg_parser
from .datasets import Dataset
from .settings import DEFAULT_CHECKPOINT_FILENAME


def main():
    """
    Entry point for the main script.

    Parses command-line arguments, initializes a neural network, processes the dataset, and either trains a new network
    or resumes training from a checkpoint.

    Prints CLI arguments and saves the trained network and loss plot.
    """
    parser = make_main_arg_parser()
    args = parser.parse_args()

    # Extract CLI arguments.
    hidden_size = args.hidden_size[0]
    learning_rate = args.learning_rate[0]
    momentum = args.momentum[0]
    num_epochs = args.num_epochs[0]

    print(
        f"CLI args: "
        f"hidden_size={hidden_size}; "
        f"learning_rate={learning_rate}; "
        f"momentum={momentum}; "
        f"num_epochs={num_epochs}"
    )

    dataset = Dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = dataset.process()

    # Initialize or load the neural network:
    # If a checkpoint file exists, load the network from the checkpoint;
    # otherwise, initialize a new network with specified architecture and hyperparameters.
    if Path(DEFAULT_CHECKPOINT_FILENAME).is_file():
        network = networks.Sequential.load(DEFAULT_CHECKPOINT_FILENAME)
    else:
        network = networks.Sequential(
            layers.Linear(2, hidden_size, activation=activations.Sigmoid(), random_state=42),
            layers.Linear(hidden_size, 2, random_state=43),
            feature_scaler=dataset.feature_scaler,
            learning_rate=learning_rate,
            momentum=momentum,
            num_epochs=num_epochs,
            output_scaler=dataset.output_scaler,
        )

    try:
        network.train(x_train, y_train, x_validation, y_validation)
    except KeyboardInterrupt:
        ...

    # Save network and loss plot.
    network.save(DEFAULT_CHECKPOINT_FILENAME)
    network.save_loss_plot()
