from .activation import Sigmoid
from .layer import Linear
from .network import Sequential


def main():
    net = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 2),
        Sigmoid(),
    )

    x = [
        (0.5, 0.3),
        (0.7, 0.5),
        (0.1, 0.8),
    ]
    y = [
        (0.6, 0.4),
        (0.8, 0.6),
        (0.2, 0.1),
    ]

    net.fit(x, y, learning_rate=0.1)
