import json

# from .dqn import DQNRunner
# from .data_loader import DataLoader
# from .arg_parser import make_arg_parser
from . import activations, layers, networks


def main():
    # parser = make_arg_parser()
    # args = parser.parse_args()
    # source_filename = args.filename[0]
    #
    learning_rate = 0.1
    momentum = 0.9
    num_epochs = 1000
    #
    # data_loader = DataLoader(
    #     source_filename=source_filename,
    # )
    # # train_loader, test_loader = data_loader.get_dataloaders()
    # x_train, y_train, x_test, y_test = data_loader.train_test_split()
    x_train, y_train = [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]
    # x_train, y_train = [[1, 0], [0, 1]], [[1], [1]]
    # x_train, y_train = [[0, 0], [1, 0]], [[0], [1]]
    # x_train, y_train = [[1, 0]], [[1]]

    # runner = DQNRunner(
    #     learning_rate=learning_rate,
    #     momentum=momentum,
    #     num_epochs=num_epochs,
    # )
    #
    # try:
    #     runner.train(train_loader, test_loader)
    # except KeyboardInterrupt:
    #     runner.save_model_state()

    network = networks.Sequential(
        layers.Linear(2, 4, activation=activations.Sigmoid()),
        layers.Linear(4, 1),
        learning_rate=learning_rate,
        momentum=momentum,
    )

    try:
        network.train(x_train, y_train, epochs=num_epochs)

        for x, y in zip(x_train, y_train):
            print(x, y, network(x))
        # network.save('ffn_checkpoint.json')
    except KeyboardInterrupt:
        # network.save('ffn_checkpoint.json')
        ...
