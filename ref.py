import numpy as np


# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# XOR data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
actual_output = np.array([[0], [1], [1], [0]])

# Neural Network Architecture
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
momentum = 0.9
epochs = 122

# Randomly initialize weights, biases, and momentum terms
weights_input_hidden = np.array([
    [0.91300363, 0.94700325, 0.75501112, 0.76566442],
    [0.85213534, 0.88962154, 0.8392711,  0.60951791],
])
biases_hidden = np.zeros((1, hidden_size))
momentum_weights_input_hidden = np.zeros_like(weights_input_hidden)
momentum_biases_hidden = np.zeros_like(biases_hidden)

weights_hidden_output = np.array([
    [0.33312217],
    [0.10090748],
    [0.47717155],
    [0.42678563],
])
biases_output = np.zeros((1, output_size))
momentum_weights_hidden_output = np.zeros_like(weights_hidden_output)
momentum_biases_output = np.zeros_like(biases_output)

# Training Loop
for epoch in range(epochs):
    epoch_loss = 0  # Accumulator for the epoch's total loss

    # Stochastic Gradient Descent with Momentum (Online Learning)
    for i in range(len(inputs)):
        # Forward Pass
        hidden_activations = sigmoid(np.dot(inputs[i], weights_input_hidden) + biases_hidden)
        predicted_output = np.dot(hidden_activations, weights_hidden_output) + biases_output

        # Loss Calculation (MSE)
        loss = 0.5 * np.mean((predicted_output - actual_output[i]) ** 2)
        epoch_loss += loss  # Accumulate loss for the epoch

        # Backward Pass
        output_error_contribution = predicted_output - actual_output[i]
        hidden_error_contribution = output_error_contribution.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_activations)

        # Weight and Bias Updates with Momentum
        momentum_weights_hidden_output = momentum * momentum_weights_hidden_output - learning_rate * hidden_activations.reshape(-1, 1).dot(output_error_contribution.reshape(1, -1))
        weights_hidden_output += momentum_weights_hidden_output
        momentum_biases_output = momentum * momentum_biases_output - learning_rate * output_error_contribution
        biases_output += momentum_biases_output

        momentum_weights_input_hidden = momentum * momentum_weights_input_hidden - learning_rate * inputs[i].reshape(-1, 1).dot(hidden_error_contribution.reshape(1, -1))
        weights_input_hidden += momentum_weights_input_hidden
        momentum_biases_hidden = momentum * momentum_biases_hidden - learning_rate * hidden_error_contribution
        biases_hidden += momentum_biases_hidden

    # Print loss for every 1 epoch
    if epoch % 1 == 0:
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.6f}")

# Test the model on XOR inputs
for test_input in inputs:
    hidden_activations = sigmoid(np.dot(test_input, weights_input_hidden) + biases_hidden)
    predicted_output = np.dot(hidden_activations, weights_hidden_output) + biases_output
    print(f"Input: {test_input}, Prediction: {predicted_output}")
