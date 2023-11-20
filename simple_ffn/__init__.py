from .activation import Sigmoid
from .layer import Layer
from .network import Network
from .preprocessor import Preprocessor


def main():
    net = Network(
        Layer(2, 8, activation=Sigmoid()),
        Layer(8, 2, activation=Sigmoid()),
        learning_rate=0.05,
    )

    pp = Preprocessor()
    x_train, y_train, x_test, y_test = pp.process()

    # learn
    net.fit(x_train, y_train, epochs=100)

    # save
    net.save()

    # predict
    # net.load('checkpoint.json')
    #
    # for i in x_train[:5]:
    #     print(i, net.predict(i))
