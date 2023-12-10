# Feed-Forward Neural Network with Backpropagation

This repository contains a pure Python implementation of a Feed-Forward neural network with Backpropagation.

## Installation and Usage

### Pre-requisites

Before getting started, ensure you have the following installed on your system:

- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/) (3.9+)

### Clone Repository

Clone this repository and navigate to the clone location:

```shell
git clone https://github.com/mezbaul-h/simple_ffn.git
cd simple_ffn
```

### Install Dependencies

Use a Python virtual environment to avoid contaminating the global package space. Install dependencies from requirements.txt:

```shell
pip install -r requirements.txt
```

### Run Neural Network

You can run the network with desired hyperparameters as a module from the command line:

```shell
python -m simple_ffn -e 10 -lr 0.01 -m 0.9
```

### Run Lander Game

To run the lander game, use the following command:

```shell
python -m lander
```


## Dataset Preprocessing

You can run the preprocessing script as follows:

```shell
python scripts/preprocess_dataset.py
```

The preprocessing script performs the following:

- Removes rows with zero values in any of the columns.
- Splits the dataset into train (70%), test (20%), and validation (10%).
- Scales only the training dataset and saves the scaling parameters for later use by the network.
- Saves these datasets and scaling parameters to appropriate files.


## Grid Search (Hyperparameter Tuning)

Initiate the grid search with the following command:

```shell
python scripts/grid_search.py
```

Grid search finds the best parameters evaluated on the final test set and saves the results in the [data/grid_result](data/grid_result) directory.
