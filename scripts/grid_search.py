"""
Grid Search Script

This script performs a grid search over specified hyperparameter space for a neural network using the simple_ffn
library. It utilizes multiprocessing to parallelize the search process.
"""
import functools
import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from simple_ffn import activations, layers, networks
from simple_ffn.datasets import Dataset

_GRID_RESULT_ROOT = PROJECT_ROOT / "data" / "grid_result"
_NUM_CPU_THREADS = os.cpu_count() or 1


def perform_search(indexed_param_dict, dataset, x_train, x_validation, x_test, y_train, y_validation, y_test):
    """
    Perform a grid search iteration for a given set of hyperparameters.

    Parameters
    ----------
    indexed_param_dict : tuple
        A tuple containing combination index and hyperparameter dictionary.
    dataset : Dataset
        The dataset object.
    x_train, x_validation, x_test, y_train, y_validation, y_test : list
        Dataset splits.

    Returns
    -------
    dict
        Dictionary containing results for the current hyperparameter combination.
    """
    combination_index, param_dict = indexed_param_dict
    hidden_size = param_dict["hidden_size"]
    network = networks.Sequential(
        layers.Linear(2, hidden_size, activation=activations.Sigmoid(), random_state=42),
        layers.Linear(hidden_size, 2, random_state=43),
        feature_scaler=dataset.feature_scaler,
        learning_rate=param_dict["learning_rate"],
        momentum=param_dict["momentum"],
        num_epochs=param_dict["epochs"],
        output_scaler=dataset.output_scaler,
    )

    network.train(x_train, y_train, x_validation, y_validation, log_prefix=f"[{combination_index + 1}] ")
    network.save_loss_plot(_GRID_RESULT_ROOT / f"{combination_index}_loss_plot.png")

    # Calculate loss on the final test set using the best layer parameters (best weights and biases at the epoch with
    # the lowest validation loss).
    avg_evaluation_losses = network.evaluate(x_test, y_test, use_best_layer_params=True)
    mean_evaluation_loss = sum(avg_evaluation_losses) / len(avg_evaluation_losses)

    return {
        "id": combination_index,
        "mean_evaluation_loss": mean_evaluation_loss,
        "param_dict": param_dict,
    }


def main():
    """
    Main function to perform grid search over specified hyperparameter space.
    """
    best_score = float("inf")
    best_param_set_id = None

    dataset = Dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = dataset.process()

    # Define hyperparameter search space
    param_grid = {
        "epochs": [300],
        "hidden_size": [2, 4, 8, 16],
        "learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "momentum": [0.5, 0.9, 0.99],
    }
    param_combinations = itertools.product(*param_grid.values())
    param_dicts = [dict(zip(param_grid.keys(), param_combination)) for param_combination in param_combinations]

    # Define a partial function for parallel grid search
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

    print(f"Worker Count: {_NUM_CPU_THREADS}")

    # Perform parallel grid search using multiprocessing Pool.
    with Pool(_NUM_CPU_THREADS) as pool:
        grid_search_results = pool.map(partial_perform_search, enumerate(param_dicts))

    # Find the best-performing hyperparameter set.
    for item in grid_search_results:
        if item["mean_evaluation_loss"] < best_score:
            best_score = item["mean_evaluation_loss"]
            best_param_set_id = item["id"]

    # Save grid search results to a JSON file.
    with open(_GRID_RESULT_ROOT / "grid_search_results.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "grid_search_results": grid_search_results,
                    "best_param_set_id": best_param_set_id,
                },
                indent=4,
            )
        )


if __name__ == "__main__":
    main()
