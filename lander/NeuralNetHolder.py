from simple_ffn import networks
from simple_ffn.settings import DEFAULT_CHECKPOINT_FILENAME, PROJECT_ROOT


class NeuralNetHolder:
    def __init__(self):
        super().__init__()

        self.network = networks.Sequential.load(PROJECT_ROOT / DEFAULT_CHECKPOINT_FILENAME)

    def predict(self, input_row: str):
        input_row = [float(item) for item in input_row.split(",")]
        scaled_input_row = self.network.feature_scaler.transform([input_row])[0]

        prediction_scaled = self.network.predict(scaled_input_row)

        prediction_unscaled = self.network.output_scaler.inverse_transform([prediction_scaled])[0]

        return prediction_unscaled[1], prediction_unscaled[0]
