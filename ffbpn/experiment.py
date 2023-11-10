from .activation import Sigmoid
from .layer import Layer
from .network import Network


def main():
    net = Network(
        Layer(2, 3, activation=Sigmoid()),
        Layer(3, 2, activation=Sigmoid()),
    )

    x = [
        (0.5, 0.3),
        (0.7, 0.5),
        # (0.1, 0.8),
    ]
    y = [
        (0.6, 0.4),
        (0.8, 0.6),
        # (0.2, 0.9),
    ]
    lr = 0.1

    # learn
    net.fit(x, y, learning_rate=lr, epochs=4000)

    # save
    weights = net.save()

    # predict
    net.load(weights)

    for i in x:
        print(i, net.predict(i, lr))
