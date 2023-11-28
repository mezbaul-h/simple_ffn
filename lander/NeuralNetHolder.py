# import pathlib
#
# import numpy
# import torch
#
# from simple_ffn.data_scaler import DataScaler
# from simple_ffn.dqn import DQNRunner
#
#
# class NeuralNetHolder:
#     def __init__(self):
#         super().__init__()
#
#         self.source_data_path = pathlib.Path(__file__).resolve().parent / "ce889_dataCollection.csv"
#         self.data_scaler = DataScaler(source_filename=str(self.source_data_path))
#
#         self.data_scaler.load_scaler_params()
#
#         self.net = DQNRunner(
#             learning_rate=0.01,
#             momentum=0.5,
#             num_epochs=500,
#         )
#
#         self.net.eval()
#
#     def predict(self, input_row: str):
#         input_row = [float(item) for item in input_row.split(',')]
#         scaled_input_row = self.data_scaler.scale_data(
#             numpy.pad(input_row, (0, 2)).reshape(1, -1)
#         )[0, :2]
#
#         prediction = self.net.model(torch.tensor(input_row).to(device='cpu', dtype=torch.float)).tolist()
#         prediction_scaled = self.net.model(torch.tensor(scaled_input_row).to(device='cpu', dtype=torch.float)).tolist()
#         prediction_unscaled = self.data_scaler.unscale_data(
#             numpy.pad(prediction_scaled, (2, 0)).reshape(1, -1)
#         )[0, 2:]
#
#         print(input_row, prediction, prediction_unscaled)
#
#         return prediction_unscaled[1], prediction_unscaled[0]
import json
import pathlib

import numpy

from simple_ffn.data_scaler import DataScaler
from simple_ffn.network2 import forward_pass


class NeuralNetHolder:
    def __init__(self):
        super().__init__()

        self.source_data_path = pathlib.Path(__file__).resolve().parent / "ce889_dataCollection.csv"
        self.data_scaler = DataScaler(source_filename=str(self.source_data_path))

        self.data_scaler.load_scaler_params()

        with open('ffn_checkpoint.json', 'r') as f:
            self.model_state = json.loads(f.read())

    def predict(self, input_row: str):
        input_row = [float(item) for item in input_row.split(',')]
        scaled_input_row = self.data_scaler.scale_data(
            numpy.pad(input_row, (0, 2)).reshape(1, -1)
        )[0, :2]

        # _, prediction = forward_pass(input_row, self.model_state['trained_weights_hidden'], self.model_state['trained_biases_hidden'], self.model_state['trained_weights_output'], self.model_state['trained_biases_output'])
        _, prediction_scaled = forward_pass(scaled_input_row, self.model_state['trained_weights_hidden'],
                                     self.model_state['trained_biases_hidden'],
                                     self.model_state['trained_weights_output'],
                                     self.model_state['trained_biases_output'])
        # prediction = prediction[0]
        prediction_scaled = prediction_scaled[0]

        prediction_unscaled = [
            item
            for item in self.data_scaler.unscale_data(
                numpy.pad(prediction_scaled, (2, 0)).reshape(1, -1)
            )[0, 2:]
        ]

        print(input_row, prediction_unscaled)

        return prediction_unscaled[1], prediction_unscaled[0]
