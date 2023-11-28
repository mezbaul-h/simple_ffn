from param_chooser.arg_parser import make_arg_parser
from simple_ffn import DataLoader, layers, activations
from simple_ffn.networks import Sequential
import itertools


def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    source_filename = args.filename[0]

    best_score = float('-inf')
    best_params = None

    data_loader = DataLoader(
        source_filename=source_filename,
    )
    x_train, y_train, x_test, y_test = data_loader.train_test_split()

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'momentum': [0.1, 0.5, 0.9],
        'hidden_size': [2, 4, 8, 16],
        'epochs': [100, 500, 1000],
    }

    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        hidden_size = param_dict['hidden_size']
        network = Sequential(
            layers.Linear(2, hidden_size, activation=activations.Sigmoid()),
            layers.Linear(hidden_size, 2, activation=None),
            learning_rate=param_dict['learning_rate'],
            momentum=param_dict['momentum'],
        )

        network.fit(x_train, y_train, epochs=param_dict['epochs'])

        score = network.get_score(x_test, y_test)

        if score > best_score:
            best_score = score
            best_params = param_dict

    print("Best params are:", best_params)
