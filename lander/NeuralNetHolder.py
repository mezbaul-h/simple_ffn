import typing

from simple_ffn.network import Network


class NeuralNetHolder:
    def __init__(self):
        super().__init__()

        self.net = Network()
        self.net.load('checkpoint.json')

    def predict(self, input_row: str) -> typing.Tuple[float, float]:
        x_pos, y_pos = [float(item) for item in input_row.split(',')]
        prediction = self.net.predict([x_pos, y_pos])
        x_velocity, y_velocity = prediction[0], prediction[1]

        return x_velocity, y_velocity
