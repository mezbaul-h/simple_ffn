import typing

from .activation import Sigmoid
from .layer import Linear


class Sequential:
    def __init__(self, *args: typing.Union[Linear, Sigmoid]):
        self.layers = args

    def forward(self, x, learning_rate):
        """
        Feeds forward a set of inputs.
        """
        print("FORWARD")
        for layer in self.layers:
            args = [x]

            if isinstance(layer, Sigmoid):
                args.append(learning_rate)

            x = layer(*args)
            print(layer.__class__, x, getattr(layer, "weights", None))
        print("\n\n\n")

    def loss(self, y):
        ...

    def backward(self):
        ...

    def fit(self, x, y, learning_rate=0.5, epochs=1000):
        for input_item, output_item in zip(x, y):
            self.forward(input_item, learning_rate)
            self.loss(output_item)
            self.backward()
