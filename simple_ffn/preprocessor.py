import json
import sys

import pandas

from .constants import MODULE_ROOT_DIR
import csv

from .util import scale_matrix


class Preprocessor:
    COLUMN_MINS = [-1920, -1080, -8, -8]
    COLUMN_MAXES = [-1 * item for item in COLUMN_MINS]

    def __init__(self, **kwargs):
        self.df = pandas.read_csv(MODULE_ROOT_DIR / '../lander/ce889_dataCollection.csv', names=["x_target", "y_target", "y_velocity", "x_velocity"])
        self.df.drop_duplicates(subset=["x_target", "y_target"], keep="last", inplace=True)
        self.df = self.df[(self.df['y_velocity'] != 0) & (self.df['x_velocity'] != 0)]

        data = self.df.values.tolist()
        data = [[float(column) for column in row] for row in data]
        data = scale_matrix(data, column_mins=self.COLUMN_MINS, column_maxes=self.COLUMN_MAXES)
        self.x = [row[:-2] for row in data]
        self.y = [row[-2:] for row in data]

    def train_test_split(self, test_size: float):
        ...

    def process(self):
        return self.x, self.y, self.x, self.y
