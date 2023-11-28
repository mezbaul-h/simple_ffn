import pathlib

import numpy

from simple_ffn.data_scaler import DataScaler
from simple_ffn.networks import Sequential


class NeuralNetHolder:
    def __init__(self):
        super().__init__()

        self.source_data_path = pathlib.Path(__file__).resolve().parent / "ce889_dataCollection.csv"
        self.data_scaler = DataScaler(source_filename=str(self.source_data_path))

        self.data_scaler.load_scaler_params()

        self.model = Sequential.load('ffn_checkpoint.json')

    def predict(self, input_row: str):
        input_row = [float(item) for item in input_row.split(',')]
        scaled_input_row = self.data_scaler.scale_data(
            numpy.pad(input_row, (0, 2)).reshape(1, -1)
        )[0, :2]

        prediction_scaled = self.model.predict(scaled_input_row)

        prediction_unscaled = [
            item
            for item in self.data_scaler.unscale_data(
                numpy.pad(prediction_scaled, (2, 0)).reshape(1, -1)
            )[0, 2:]
        ]

        return prediction_unscaled[1], prediction_unscaled[0]
