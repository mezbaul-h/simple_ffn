import operator
import sys
from functools import reduce
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

import itertools
import json
import uuid

from simple_ffn import activations, layers, networks
from simple_ffn.datasets import Dataset


def main():
    best_score = float("inf")
    best_params = None

    dataset = Dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = dataset.process()

    param_grid = {
        "epochs": [250],
        "hidden_size": [2, 4, 8, 16],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2],
        "momentum": [0.5, 0.8, 0.9],
    }
    num_combinations = reduce(operator.mul, [len(item) for item in param_grid.values()])
    grid_search_results = {}

    for combination_index, params in enumerate(itertools.product(*param_grid.values())):
        print(f'{"=" * 60} ' f"GRID SEARCH START [{combination_index + 1}/{num_combinations}]")
        param_set_id = uuid.uuid4().hex
        param_dict = dict(zip(param_grid.keys(), params))
        hidden_size = param_dict["hidden_size"]
        network = networks.Sequential(
            layers.Linear(2, hidden_size, activation=activations.Sigmoid()),
            layers.Linear(hidden_size, 2, activation=None),
            feature_scaler=dataset.feature_scaler,
            learning_rate=param_dict["learning_rate"],
            momentum=param_dict["momentum"],
            num_epochs=param_dict["epochs"],
            output_scaler=dataset.output_scaler,
        )

        network.train(x_train, y_train, x_validation, y_validation)

        avg_evaluation_losses = network.evaluate(x_test, y_test)
        mean_evaluation_loss = sum(avg_evaluation_losses) / len(avg_evaluation_losses)

        network.save_loss_plot(f"{param_set_id}_loss_plot.png")

        grid_search_results[param_set_id] = param_dict

        if mean_evaluation_loss < best_score:
            best_score = mean_evaluation_loss
            best_params = param_dict

        # Save in each iteration, so that if program shuts down during
        # execution we have something to work with.
        with open("grid_search_results.json", "w") as f:
            f.write(json.dumps(grid_search_results))

    print("Best params are:", best_params)


if __name__ == "__main__":
    main()
