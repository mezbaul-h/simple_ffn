from ffbpn.util import generate_random_vector


class Linear:
    def __init__(self, input_feature_count: int, output_feature_count: int):
        self.input_feature_count = input_feature_count
        self.output_feature_count = output_feature_count
        self.weights = generate_random_vector((input_feature_count, output_feature_count))

    def __call__(self, x):
        outputs = [0] * self.output_feature_count

        for i in range(self.output_feature_count):
            outputs[i] = sum([x[j] * self.weights[j][i] for j in range(len(x))])

        return outputs
