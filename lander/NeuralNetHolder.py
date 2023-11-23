from simple_ffn.network import Network
from simple_ffn.util import scale_min_max
from simple_ffn.preprocessor import Preprocessor


class NeuralNetHolder:
    def __init__(self):
        super().__init__()

        self.net = Network()
        self.net.load('checkpoint.json')

    def _rescale(self, val, val_min, val_max):
        return (val * (val_max - val_min)) + val_min

    def predict(self, input_row: str):
        input_row = [float(item) for item in input_row.split(',')]

        x_pos, y_pos = (
            scale_min_max(input_row[0], Preprocessor.COLUMN_MINS[0], Preprocessor.COLUMN_MAXES[0]),
            scale_min_max(input_row[1], Preprocessor.COLUMN_MINS[1], Preprocessor.COLUMN_MAXES[1]),
        )
        prediction = self.net.predict([x_pos, y_pos])
        y_velocity, x_velocity = (
            self._rescale(prediction[0], Preprocessor.COLUMN_MINS[2], Preprocessor.COLUMN_MAXES[2]),
            self._rescale(prediction[1], Preprocessor.COLUMN_MINS[2], Preprocessor.COLUMN_MAXES[2]),
        )

        # print(x_velocity, y_velocity)
        print('preds', prediction)

        return x_velocity, y_velocity
