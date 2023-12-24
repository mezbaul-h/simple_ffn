# Feed-Forward Neural Network with Backpropagation

This repository features a pure Python implementation of a Feed-Forward neural network with Backpropagation. It serves as the foundation for training a shallow deep learning model designed to play the lander game. This project constituted my final individual assignment for the course CE889: Neural Networks and Deep Learning during my master's studies at the University of Essex.


## Installation

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

### Install Python Dependencies

Use a Python virtual environment to avoid contaminating the global package space. Install dependencies from requirements.txt:

```shell
pip install -r requirements.txt
```


## Usage

### Run Neural Network

To execute the network with customized hyperparameters as a module from the command line, use the following:

```shell
python -m simple_ffn --num-epochs 10 --learning-rate 0.01 --momentum 0.9 --hidden-size 2
```

This command will store the model states in a [checkpoint](#checkpoint-file) file.

### Run Lander Game

To run the lander game, use the following command:

```shell
python -m lander
```

To employ the trained model for gameplay, select "**Neural Network**" from the menu after launching the game. Subsequently, click any button to initiate rocket movement, and the model will seamlessly handle the movements thereafter.


## Checkpoint File

The program defaults to utilizing the checkpoint file specified by the `DEFAULT_CHECKPOINT_FILENAME` constant in [simple_ffn/settings.py](simple_ffn/settings.py) for both saving (during training) and loading (for resuming training and in-game model loading) states.


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

You can initiate the grid search with the following command:

```shell
python scripts/grid_search.py
```

Grid search finds the best parameters evaluated on the final test set and saves the results in the [data/grid_result](data/grid_result) directory.
