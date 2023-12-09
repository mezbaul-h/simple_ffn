from pathlib import Path

from . import activations, layers, networks
from .arg_parsers import make_main_arg_parser
from .datasets import Dataset
from .settings import DEFAULT_CHECKPOINT_FILENAME


def main():
    parser = make_main_arg_parser()
    args = parser.parse_args()

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

    # If checkpoint exists, resume from that.
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

    network.save("ffn_checkpoint.json")
    network.save_loss_plot()
