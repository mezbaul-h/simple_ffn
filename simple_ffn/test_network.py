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
    lr = 0.1
    net = Network(
        Layer(2, 2, activation=Sigmoid()),
        Layer(2, 2, activation=None),
        learning_rate=lr,
    )
    x = [
        [0.7075148894363186, 0.6999537037037038],
        [0.7075044727696519, 0.7000925925925926]
    ]
    y = [
        [0.49375, 0.5025],
        [0.48125, 0.5025]
    ]

    net.layers[0].weights = [
        [
            0.6325858845538125,
            0.8815493308804143
        ],
        [
            0.3919195106891443,
            0.8489334826444461
        ]
    ]

    net.layers[1].weights = [
        [
            0.19826811185185192,
            0.9872763887752762
        ],
        [
            0.8884594727985691,
            0.48460053022885774
        ]
    ]

    for i in range(25):
        for inp, outp in zip(x, y):
            out = net.forward(inp)
            losses = net.loss(out, outp)
            net.backward(losses)

        print(0, net.layers[0].delta_weights)
        print(0, net.layers[0].weights)
        print(1, net.layers[1].delta_weights)
        print(1, net.layers[1].weights)
        print('--')

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
