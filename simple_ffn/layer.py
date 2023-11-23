import copy
import json
import sys

from simple_ffn.activation import Sigmoid
from simple_ffn.util import generate_random_vector


class Layer:
    def __init__(self, input_feature_count: int, output_feature_count: int, activation: Sigmoid = None):
        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.inputs = None
        self.outputs = None
        self.weights = generate_random_vector((input_feature_count, output_feature_count))
        self.delta_weights = generate_random_vector((input_feature_count, output_feature_count))
        self.activation = activation

    def update_weights(self, learning_rate):
        for i in range(self.input_feature_count):
            for j in range(self.output_feature_count):
                # self.weights[i][j] = self.weights[i][j] - (learning_rate * self.delta_weights[i][j])
                self.weights[i][j] = self.delta_weights[i][j] + self.weights[i][j]

    def __call__(self, x, learning_rate):
        self.inputs = x
        self.outputs = [0] * self.output_feature_count

        for i in range(self.output_feature_count):
            self.outputs[i] = sum([x[j] * self.weights[j][i] for j in range(len(x))])

        if self.activation:
            self.outputs = self.activation(self.outputs, learning_rate)

        return self.outputs.copy()

    def __str__(self):
        print("--- LAYER INFORMATION ---")
        print("INPUTS")
        print(json.dumps(self.inputs, indent=4))
        print("WEIGHTS")
        print(json.dumps(self.weights, indent=4))
        print("OUTPUTS")
        print(json.dumps(self.outputs, indent=4))
        print("-------------------------")
        return ''
