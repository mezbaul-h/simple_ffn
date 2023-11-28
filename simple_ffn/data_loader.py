import pandas


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

    def train_test_split(self):
        test_nd_matrix = self.test_df.to_numpy()
        x_test = test_nd_matrix[:, :-2]
        y_test = test_nd_matrix[:, -2:]

        train_nd_matrix = self.train_df.to_numpy()
        x_train = train_nd_matrix[:, :-2]
        y_train = train_nd_matrix[:, -2:]

        return x_train, y_train, x_test, y_test
