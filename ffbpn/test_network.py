import math


def test_forward_propagation(network_232):
    x = [
        [0.3, 0.5],
        [0.7, 0.1]
    ]
    y = [
        [0.3, 1],
        [0.1, 0.1],
    ]
    lr = 0.7

    for a, b in zip(x, y):
        network_232.forward(a, lr)

    output_layer = network_232.layers[-1]
    output_layer_inputs = output_layer.inputs
    output_layer_weights = output_layer.weights
    predicted_outputs = output_layer.outputs

    assert predicted_outputs[0] == 1/(1 + math.exp(-lr * sum([output_layer_inputs[i] * output_layer_weights[i][0] for i in range(output_layer.input_feature_count)])))


def test_backward_propagation(network_232):
    x = [
        [0.3, 0.5],
    ]
    y = [
        [0.3, 1],
    ]
    lr = 0.7

    for a, b in zip(x, y):
        calculated_outputs = network_232.forward(a, lr)

        losses = network_232.loss(calculated_outputs, b)

        network_232.backward(losses, lr)

    output_layer = network_232.layers[-1]
    output_layer_inputs = output_layer.inputs
    # output_layer_weights = output_layer.weights
    output_layer_delta_weights = output_layer.delta_weights
    predicted_outputs = output_layer.outputs

    exp = ((predicted_outputs[0] - y[0][0]) * (predicted_outputs[0] * (1 - predicted_outputs[0])) * output_layer_inputs[0])
    assert output_layer_delta_weights[0][0] == exp
