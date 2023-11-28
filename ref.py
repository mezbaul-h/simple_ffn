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
epochs = 1000

# Randomly initialize weights and biases
weights_input_hidden = np.array([
    [0.91300363, 0.94700325, 0.75501112, 0.76566442],
    [0.85213534, 0.88962154, 0.8392711,  0.60951791],
])
biases_hidden = np.zeros((1, hidden_size))

weights_hidden_output = np.array([
    [0.33312217],
    [0.10090748],
    [0.47717155],
    [0.42678563],
])
biases_output = np.zeros((1, output_size))

# Training Loop
for epoch in range(epochs):
    epoch_loss = 0  # Accumulator for the epoch's total loss

    # Stochastic Gradient Descent (Online Learning)
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

        # Weight and Bias Updates
        weights_hidden_output -= learning_rate * hidden_activations.reshape(-1, 1).dot(output_error_contribution.reshape(1, -1))
        biases_output -= learning_rate * output_error_contribution

        weights_input_hidden -= learning_rate * inputs[i].reshape(-1, 1).dot(hidden_error_contribution.reshape(1, -1))
        biases_hidden -= learning_rate * hidden_error_contribution

    # Print loss for every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss}")

# Test the model on XOR inputs
for test_input in inputs:
    hidden_activations = sigmoid(np.dot(test_input, weights_input_hidden) + biases_hidden)
    predicted_output = np.dot(hidden_activations, weights_hidden_output) + biases_output
    print(f"Input: {test_input}, Prediction: {predicted_output}")
