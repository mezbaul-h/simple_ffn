import json
import pathlib

import pandas
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    COLUMN_HEADERS = ["x_target", "y_target", "y_velocity", "x_velocity"]
    COLUMN_MINS = [-1920, -1080, -8, -8]
    COLUMN_MAXES = [abs(item) for item in COLUMN_MINS]

    def __init__(self, **kwargs):
        self.source_filename: str = kwargs['source_filename']
        self.filename_prefix = '.'.join(self.source_filename.split('.')[:-1])
        self.df = pandas.read_csv(
            pathlib.Path(self.source_filename),
            names=self.COLUMN_HEADERS,
        )
        self.test_df = None
        self.train_df = None

        # Initialize the MinMaxScaler
        self.scaler = MinMaxScaler()

    def sanitize_data(self):
        self.df = self.df.round(decimals=4)

        self.df.drop_duplicates(subset=["x_target", "y_target"], keep="last", inplace=True)

        self.df = self.df.apply(pandas.to_numeric)

    def save_data(self):
        self.test_df.to_csv(f'{self.filename_prefix}.test.csv', header=False, index=False)
        self.train_df.to_csv(f'{self.filename_prefix}.train.csv', header=False, index=False)

    def save_scaler_params(self):
        scaler_params = {
            'min_': self.scaler.min_.tolist(),
            'scale_': self.scaler.scale_.tolist(),
        }

        with open(f'{self.filename_prefix}.scaler_params.json', 'w+') as f:
            f.write(json.dumps(scaler_params, indent=4))

    def scale_data(self):
        # Fit and transform the data
        scaled_data = self.scaler.fit_transform(self.train_df)

        self.train_df = pandas.DataFrame(scaled_data, columns=self.COLUMN_HEADERS)

    def train_test_split(self):
        # Separate features (x) and target variables (y)
        x = self.df[self.COLUMN_HEADERS[:-2]]
        y = self.df[self.COLUMN_HEADERS[-2:]]

        # Split the data into training and testing sets
        # The test_size parameter specifies the proportion of the dataset to include in the test split
        # The random_state parameter ensures reproducibility by fixing the random seed
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

        self.test_df = x_test.join(y_test)
        self.train_df = x_train.join(y_train)
