import functools
import operator
import os
import random
import sys
from functools import reduce
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

import itertools
import json
import uuid

from simple_ffn import activations, layers, networks
from simple_ffn.datasets import Dataset
from simple_ffn.utils import train_test_split

_NUM_CPU_THREADS = os.cpu_count() or 1


def perform_search(indexed_param_dict, dataset, x_train, x_validation, x_test, y_train, y_validation, y_test):
    combination_index, param_dict = indexed_param_dict
    param_set_id = uuid.uuid4().hex
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

    network.train(x_train, y_train, x_validation, y_validation, log_prefix=f"[Grid No. {combination_index + 1}] ")

    avg_evaluation_losses = network.evaluate(x_test, y_test)
    mean_evaluation_loss = sum(avg_evaluation_losses) / len(avg_evaluation_losses)

    network.save_loss_plot(f"{param_set_id}_loss_plot.png")

    return {
        **param_dict,
        "mean_evaluation_loss": mean_evaluation_loss,
    }


def main():
    best_score = float("inf")
    best_params = None

    dataset = Dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = dataset.process()

    # Take a subset (60%) of the original test set.
    x_train, x_residue, y_train, y_residue = train_test_split(x_train, y_train, random_state=42, test_size=0.4)

    # param_grid = {
    #     "epochs": [100],
    #     "hidden_size": [2, 4, 8, 16],
    #     "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2],
    #     "momentum": [0.5, 0.8, 0.9],
    # }
    param_grid = {
        "epochs": [2],
        "hidden_size": [2, 4],
        "learning_rate": [0.0001, 0.001],
        "momentum": [0.9],
    }
    param_combinations = itertools.product(*param_grid.values())
    param_dicts = [dict(zip(param_grid.keys(), param_combination)) for param_combination in param_combinations]

    partial_perform_search = functools.partial(
        perform_search,
        dataset=dataset,
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )

    print(f"Starting grid search with a pool of {_NUM_CPU_THREADS}")

    with Pool(_NUM_CPU_THREADS) as pool:
        grid_search_results = pool.map(partial_perform_search, enumerate(param_dicts))

    for item in grid_search_results:
        if item["mean_evaluation_loss"] < best_score:
            best_score = item["mean_evaluation_loss"]
            best_params = item

    with open("grid_search_results.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "grid_search_results": grid_search_results,
                    "best_params": best_params,
                },
                indent=4,
            )
        )


if __name__ == "__main__":
    main()
