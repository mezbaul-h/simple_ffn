import typing
import math
import random


class Neuron:
    def __init__(self, value: int) -> None:
        self.value = value

    def activate(self, neuron_index: int, previous_layer: typing.List, weight_matrix, learning_rate: float):
        v = []

        for i in range(len(previous_layer)):
            previous_layer_neuron = previous_layer[i]
            v.append(previous_layer_neuron.value * weight_matrix[i][neuron_index])

        v = sum(v)
        self.value = self.apply_sigmoid(v, learning_rate)

    def apply_sigmoid(self, v, learning_rate):
        new_value = 1 / (1 + math.exp(-learning_rate * v))
        return new_value

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return str(self.value)


class FFNN:
    def __init__(self, input_layer: typing.List[Neuron], expected_outputs=None, output_layer_size=None,
                 hidden_layer_size=None, hidden_layer_count=None, learning_rate=None) -> None:
        self.input_layer = input_layer

        self.expected_outputs = expected_outputs

        # number of input neurons
        self.input_size = len(input_layer)

        # lambda
        self.learning_rate = learning_rate

        # number of hidden layers
        self.hidden_layer_count = hidden_layer_count

        # number of output neurons
        self.output_layer_size = output_layer_size

        # number of neurons per hidden layer
        self.hidden_layer_size = hidden_layer_size

        self.layers = self.generate_layers()
        self.layer_count = len(self.layers)

        self.weight_matrices = self.generate_weight_matrices()

    def generate_layers(self):
        layers = [self.input_layer]

        for _ in range(self.hidden_layer_count):
            layer = []

            for _ in range(self.hidden_layer_size):
                layer.append(Neuron(0))

            layers.append(layer)

        out_layer = []

        for _ in range(self.output_layer_size):
            out_layer.append(Neuron(0))

        layers.append(out_layer)

        return layers

    def get_random_weight(self) -> float:
        return random.random()

    def get_random_weight_vector(self, vector_size: int) -> typing.List[float]:
        weight_vector = []

        for _ in range(vector_size):
            weight_vector.append(self.get_random_weight())

        return weight_vector

    def generate_weight_matrices(self):
        wms = []

        for i in range(self.hidden_layer_count + 1):
            previous_layer = self.layers[i]
            current_layer = self.layers[i + 1]
            wm = []

            for _ in range(len(previous_layer)):
                wm.append(self.get_random_weight_vector(len(current_layer)))

            wms.append(wm)

        return wms

    def forward(self):
        for i in range(self.hidden_layer_count + 1):
            previous_layer = self.layers[i]
            current_layer = self.layers[i + 1]
            weight_matrix = self.weight_matrices[i]

            for neuron_index in range(len(current_layer)):
                neuron = current_layer[neuron_index]
                neuron.activate(neuron_index, previous_layer, weight_matrix, self.learning_rate)

        gradients = self.calculate_gradients()

        # print(gradients)

    def calculate_gradients(self):
        gradients = []

        for i in range(self.layer_count - 1, -1, -1):
            current_layer = self.layers[i]
            sub_gradients = []

            if i == self.layer_count - 1:
                # output layer
                for neuron, expected_output in zip(current_layer, self.expected_outputs):
                    gradient = neuron.value * (1 - neuron.value) * (expected_output - neuron.value)
                    sub_gradients.append(gradient)
            else:
                # hidden layer
                for neuron in current_layer:
                    gradient = neuron.value * (1 - neuron.value) * ()

            gradients.append(sub_gradients)

        gradients.reverse()

        return gradients

    def print_network(self):
        print("WEIGHTS")
        print(self.weight_matrices)
        print("\n\nNETWORK")
        for layer in self.layers:
            print([neuron.value for neuron in layer])


def main():
    neuron1 = Neuron(0.1)
    neuron2 = Neuron(0.5)

    nn = FFNN(
        [neuron1, neuron2],
        expected_outputs=[0.3, 0.6],
        output_layer_size=2,
        hidden_layer_size=3,
        hidden_layer_count=1,
        learning_rate=0.8,
    )

    nn.forward()

    # nn.print_network()


if __name__ == "__main__":
    main()
