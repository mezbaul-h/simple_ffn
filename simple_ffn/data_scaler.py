import json

from sklearn.preprocessing import MinMaxScaler


class DataScaler:
    COLUMN_HEADERS = ["x_target", "y_target", "y_velocity", "x_velocity"]
    COLUMN_MINS = [-1920, -1080, -8, -8]
    COLUMN_MAXES = [abs(item) for item in COLUMN_MINS]

    def __init__(self, **kwargs):
        self.source_filename: str = kwargs['source_filename']
        self.filename_prefix = '.'.join(self.source_filename.split('.')[:-1])

        # Initialize the MinMaxScaler
        self.scaler = MinMaxScaler()

    def load_scaler_params(self):
        with open(f'{self.filename_prefix}.scaler_params.json', 'r') as f:
            scaler_params = json.loads(f.read())

        # Initialize the MinMaxScaler with the saved parameters
        self.scaler.min_ = scaler_params['min_']
        self.scaler.scale_ = scaler_params['scale_']

    def scale_data(self, data):
        # Fit and transform the data
        return self.scaler.transform(data)

    def unscale_data(self, data):
        # Fit and transform the data
        return self.scaler.inverse_transform(data)
