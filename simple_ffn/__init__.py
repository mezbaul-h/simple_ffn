from .data_loader import DataLoader
from .arg_parser import make_arg_parser
from . import activations, layers, networks


def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    source_filename = args.filename[0]

    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 100

    data_loader = DataLoader(
        source_filename=source_filename,
    )
    x_train, y_train, x_test, y_test = data_loader.train_test_split()

    network = networks.Sequential(
        layers.Linear(2, 4, activation=activations.Sigmoid()),
        layers.Linear(4, 2),
        learning_rate=learning_rate,
        momentum=momentum,
    )

    try:
        network.train(x_train, y_train, epochs=num_epochs)

        # network.save('ffn_checkpoint.json')
    except KeyboardInterrupt:
        # network.save('ffn_checkpoint.json')
        ...
