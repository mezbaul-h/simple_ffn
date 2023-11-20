import math

from . import Network, Layer
from .activation import Sigmoid


# def test_forward_propagation(network_232):
#     x = [
#         [0.3, 0.5],
#         [0.7, 0.1]
#     ]
#     y = [
#         [0.3, 1],
#         [0.1, 0.1],
#     ]
#
#     for a, b in zip(x, y):
#         network_232.forward(a)
#
#     output_layer = network_232.layers[-1]
#     output_layer_inputs = output_layer.inputs
#     output_layer_weights = output_layer.weights
#     predicted_outputs = output_layer.outputs
#
#     assert predicted_outputs[0] == Sigmoid.calculate_sigmoid(sum([output_layer_inputs[i] * output_layer_weights[i][0] for i in range(output_layer.input_feature_count)]), network_232.learning_rate)


def test_forward_propagation2():
    lr = 1.0
    net = Network(
        Layer(2, 2, activation=Sigmoid()),
        Layer(2, 1, activation=Sigmoid()),
        learning_rate=lr,
    )
    x = [
        [0.35, 0.9],
    ]
    y = [
        [0.5],
    ]

    net.layers[0].weights = [
        [0.1, 0.4],
        [0.8, 0.6],
    ]
    net.layers[1].weights = [
        [0.3],
        [0.9],
    ]

    out = net.forward(x[0])
    losses = net.loss(out, y[0])

    net.backward(losses)

    print(net.layers[0].weights)
    print(net.layers[1].weights)

    raise

# def test_backward_propagation(network_232):
#     x = [
#         [0.3, 0.5],
#     ]
#     y = [
#         [0.3, 1],
#     ]
#
#     for a, b in zip(x, y):
#         calculated_outputs = network_232.forward(a)
#
#         losses = network_232.loss(calculated_outputs, b)
#
#         network_232.backward(losses)
#
#     output_layer = network_232.layers[-1]
#     output_layer_inputs = output_layer.inputs
#     # output_layer_weights = output_layer.weights
#     output_layer_delta_weights = output_layer.delta_weights
#     predicted_outputs = output_layer.outputs
#
#     exp = ((predicted_outputs[0] - y[0][0]) * (predicted_outputs[0] * (1 - predicted_outputs[0])) * output_layer_inputs[0])
#     assert output_layer_delta_weights[0][0] == exp
