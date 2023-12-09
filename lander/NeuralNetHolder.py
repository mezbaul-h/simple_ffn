import math

from simple_ffn.networks import Sequential
from simple_ffn.settings import PROJECT_ROOT


class NeuralNetHolder:
    def __init__(self):
        super().__init__()

        self.model = Sequential.load(PROJECT_ROOT / "ffn_checkpoint.json")

    def predict(self, input_row: str):
        input_row = [float(item) for item in input_row.split(",")]
        scaled_input_row = self.model.feature_scaler.transform([input_row])[0]

        prediction_scaled = self.model.predict(scaled_input_row)

        prediction_unscaled = self.model.output_scaler.inverse_transform([prediction_scaled])[0]

        return prediction_unscaled[1], prediction_unscaled[0]
