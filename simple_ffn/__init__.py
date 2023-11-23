from .activation import Sigmoid
from .layer import Layer
from .network import Network
from .preprocessor import Preprocessor


def main():
    net = Network(
        Layer(2, 2, activation=Sigmoid()),
        Layer(2, 2, activation=None),
        learning_rate=0.001,
    )

    pp = Preprocessor()
    x_train, y_train, x_test, y_test = pp.process()

    # learn
    net.fit(x_train, y_train, epochs=25)

    # save
    net.save()

    # predict
    # net.load('checkpoint.json')
    #
    # for i in x_train:
    #     print(i, net.predict(i))
