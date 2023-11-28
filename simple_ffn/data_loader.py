import pandas

import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset


class DataLoader:
    COLUMN_HEADERS = ["x_target", "y_target", "y_velocity", "x_velocity"]
    COLUMN_MINS = [-1920, -1080, -8, -8]
    COLUMN_MAXES = [abs(item) for item in COLUMN_MINS]

    def __init__(self, **kwargs):
        self.source_filename: str = kwargs['source_filename']
        self.filename_prefix = '.'.join(self.source_filename.split('.')[:-1])
        self.test_df = pandas.read_csv(
            f'{self.filename_prefix}.test.csv',
            names=self.COLUMN_HEADERS,
        )
        self.train_df = pandas.read_csv(
            f'{self.filename_prefix}.train.csv',
            names=self.COLUMN_HEADERS,
        )

    def get_dataloaders(self):
        x_train, y_train, x_test, y_test = self.train_test_split()
        x_train, y_train, x_test, y_test = self.to_tensors(x_train, y_train, x_test, y_test)
        train_loader, test_loader = self.make_dataloaders(x_train, y_train, x_test, y_test)

        return train_loader, test_loader

    @staticmethod
    def make_dataloaders(x_train, y_train, x_test, y_test):
        # Create DataLoader for training and testing sets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        batch_size = 64
        train_loader = TorchDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = TorchDataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def to_tensors(x_train, x_test, y_train, y_test):
        # Convert data to PyTorch tensors
        return [
            torch.tensor(item).to(device="cpu", dtype=torch.float) for item in (x_train, x_test, y_train, y_test)
        ]

    def train_test_split(self):
        test_nd_matrix = self.test_df.to_numpy()
        x_test = test_nd_matrix[:, :-2]
        y_test = test_nd_matrix[:, -2:]

        train_nd_matrix = self.train_df.to_numpy()
        x_train = train_nd_matrix[:, :-2]
        y_train = train_nd_matrix[:, -2:]

        return x_train, y_train, x_test, y_test
