import json

from .constants import MODULE_ROOT_DIR
import csv

from .util import scale_matrix


class Preprocessor:
    def __init__(self, **kwargs):
        with open(MODULE_ROOT_DIR / '../lander/ce889_dataCollection.csv', 'r') as f:
            r = csv.reader(f)
            data = [[float(column) for column in row] for row in r if row]
            data = scale_matrix(data)
            self.x = [[row[0], row[1]] for row in data]
            self.y = [[row[2], row[3]] for row in data]

    def train_test_split(self, test_size: float):
        ...

    def process(self):
        return self.x, self.y, self.x, self.y
