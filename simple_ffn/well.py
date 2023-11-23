"Collection of activations functions for the neural network"

import numpy as np


# define some activation and error functions
def tanh(x, derivative=False):
    """Implements the hyperbolic tangent function element wise over an array x.

    Parameters
    ----------
    x : numpy array
        This array contains arguments for the hyperbolic tangent function.
    derivative : bool
        Indicates whether to use the hyperbolic tangent function or its derivative.

    Returns
    -------
    numpy array
        An array of equal shape to `x`.
    """

    if derivative:
        tanh_not_derivative = tanh(x)
        return 1.0 - tanh_not_derivative ** 2
        # return 1.0 - x**2
    else:
        return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        return 1 * (x > 0)  # returns 1 for any x > 0, and 0 otherwise

    return np.maximum(0, x)


def softmax(x, derivative=False):
    if derivative:
        pass

    return np.exp(x) / sum(np.exp(x))


def softmax_grad(softmax):
    # https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
    # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
    #

    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def mean_squared_error(target_output, actual_output, derivative=False):
    try:
        assert (target_output.shape == actual_output.shape)
    except AssertionError:
        print(
            f"Shape of target vector: {target_output.shape} does not match shape of actual vector: {actual_output.shape}")
    if derivative:
        error = (actual_output - target_output)

    else:
        error = np.sum(0.5 * np.sum((target_output - actual_output) ** 2, axis=1, keepdims=True))

    return error


class NeuralNet(object):
    RNG = np.random.default_rng()

    def __init__(self,
                 topology: list[int] = [],
                 learning_rate=0.01,
                 momentum=0.1,
                 hidden_activation_func=relu,
                 output_activation_func=tanh,
                 init_method='random'):

        self.topology = topology
        self.weight_mats = []
        self.bias_mats = []  # will hold the weights for the bias nodes

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.hidden_activation = hidden_activation_func
        self.output_activation = output_activation_func

        self._init_weights_and_biases(init_method)
        self.size = len(self.weight_mats)
        self.netIns = [None] * self.size  # store the inputs to each layer
        self.netOuts = [None] * self.size  # store the activations from each layer
        # self.stored_gradients = [None] * self.size
        self.last_change = [np.zeros(mat.shape) for mat in self.weight_mats]

        # -- create similar lists to store gradients for the bias weigths
        # self.stored_bias_gradients = [np.zeros(mat.shape) for mat in self.bias_mats]
        self.last_bias_change = [np.zeros(mat.shape) for mat in self.bias_mats]

    def _init_weights_and_biases(self, method='random'):
        # -- decide which initialization method to use. I added some of the popular ones
        if method.lower() == 'random':
            _init_func = lambda num_rows, num_cols: self.RNG.random(size=(num_rows, num_cols))

        elif method.lower() == 'xavier':
            _init_func = self._xavier_weight_initialization

        else:
            print(f"\t-> initialization method {method} not recognized. Defaulting to 'random'")
            _init_func = lambda num_rows, num_cols: self.RNG.random(size=(num_rows, num_cols))

        # -- set up matrices
        if len(self.topology) > 1:
            j = 1
            for i in range(len(self.topology) - 1):
                num_rows = self.topology[i]
                num_cols = self.topology[j]

                mat = _init_func(num_rows, num_cols)  # the +1 accounts for the bias weights
                bias_vector = _init_func(1, num_cols)

                self.weight_mats.append(mat)
                self.bias_mats.append(bias_vector)

                j += 1

    def _xavier_weight_initialization(self, num_rows, num_cols):
        '''A type of weight initialization that seems to be tailored to sigmoidal activation functions.
        Here is a reference: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/'''
        num_inputs = self.topology[0]

        lower_bound = -1 / np.sqrt(num_inputs)
        upper_bound = 1 / np.sqrt(num_inputs)

        mat = self.RNG.uniform(lower_bound, upper_bound, (num_rows, num_cols))
        return mat

    @property
    def shape(self):
        return tuple(self.topology)

    @property
    def n_trainable_params(self):
        n_params = 0
        for weight_mat, bias_mat in zip(self.weight_mats, self.bias_mats):
            n_params += weight_mat.size + bias_mat.size

        return n_params

    def feedforward(self, input_vector):

        self.netIns.clear()
        self.netOuts.clear()

        I = input_vector  # rename vector to match typical nomenclature

        for idx, W in enumerate(self.weight_mats):

            bias_vector = self.bias_mats[idx]

            self.netOuts.append(I)  # storing activations from the last layer
            I = np.dot(I, W) + bias_vector
            self.netIns.append(I)  # storing the inputs to the current layer

            # -- apply activation function
            if idx == len(self.weight_mats) - 1:
                out_vector = self.output_activation(I)  # output layer
            else:
                I = self.hidden_activation(I)  # hidden layers

        return out_vector

    def _gradient_descent(self, layer_idx, gradient_mat, bias_gradient):

        # -- Calculate the changes in weights for all nodes and bias weights
        delta_weight = (self.momentum * self.last_change[layer_idx]) - (self.learning_rate * gradient_mat[layer_idx])

        delta_bias_weights = (self.momentum * self.last_bias_change[layer_idx]) \
                             - (self.learning_rate * bias_gradient)

        # -- Update Weights
        self.weight_mats[layer_idx] += delta_weight  # the negative of the gradient is above
        self.bias_mats[layer_idx] += delta_bias_weights

        # -- Keep track of latest delta weights for next iteration/epoch to compute momentum term
        self.last_change[layer_idx] = 1 * delta_weight
        self.last_bias_change[layer_idx] = 1 * delta_bias_weights

    def backprop(self,
                 target,
                 output,
                 error_func, ):
        """Backpropagation.

        Parameters
        ----------
        target : numpy array
            Matching targets for each sample in `input_samples`.
        output : numpy array
            Actual output from feedforward propagation. It will be used to check the network's error.
        error_func : function object
            This is the function that computes the error of the epoch and used during backpropagation.
            It must accept parameters as: error_func(target={target numpy array},actual={actual output from network},derivative={boolean to indicate operation mode})
        """

        # Compute gradients and deltas
        for i in range(self.size):
            back_index = self.size - 1 - i  # This will be used for the items to be accessed backwards

            if i == 0:  # final layer
                d_activ = self.output_activation(self.netIns[back_index], derivative=True)
                d_error = error_func(target, output, derivative=True)
                delta = d_error * d_activ  # this should be the hadamard product, I think
                # delta = np.multiply(d_error, d_activ)

                gradient_mat = np.dot(self.netOuts[back_index].T, delta)
                bias_grad_mat = 1 * delta

                # -- Apply gradient descent
                self._gradient_descent(layer_idx=back_index, gradient_mat=gradient_mat, bias_gradient=bias_grad_mat)

            else:  # hidden layers
                W_trans = self.weight_mats[back_index + 1].T  # we use the transpose of the weights in the current layer
                d_activ = self.hidden_activation(self.netIns[back_index], derivative=True)  # δl=((wl+1)Tδl+1)⊙σ′(zl)
                d_error = np.dot(delta, W_trans)
                delta = d_error * d_activ  # this should be the hadamard product, I think
                # delta = np.multiply(d_error, d_activ)

                gradient_mat = np.dot(self.netOuts[back_index].T, delta)
                bias_grad_mat = 1 * delta

                # -- Apply gradient descent
                self._gradient_descent(layer_idx=back_index, gradient_mat=gradient_mat, bias_gradient=bias_grad_mat)

    def train(self, input_set, target_set, epochs=1000, batch_size=0, error_threshold=1E-10,
              error_func=mean_squared_error, verbose=True):

        if batch_size == 0:  # online training (one sample at a time)

            for epoch in range(epochs):
                error = 0

                for i in range(len(input_set)):
                    inputs = input_set[
                             i:i + 1]  # slicing it this way makes sure that the resulting numpy array maintains all of its dimensions
                    targets = target_set[i:i + 1]

                    error += self._train_helper(inputs, targets, error_func)

                if verbose and (epoch % 20 == 0):
                    self._print_training_info(epoch, epochs, error, error_threshold)

                if error <= error_threshold:
                    print(f"\t-> error {error} is lower than threshold {error_threshold}\n\tStopped at epoch {epoch}")
                    break

        elif batch_size == -1:  # batch training (use full training set)

            for epoch in range(epochs):
                error = 0

                inputs = input_set
                targets = target_set

                error += self._train_helper(inputs, targets, error_func)

                if verbose and (epoch % 20 == 0):
                    self._print_training_info(epoch, epochs, error, error_threshold)

                if error <= error_threshold:
                    print(f"\t-> error {error} is lower than threshold {error_threshold}\n\tStopped at epoch {epoch}")
                    break

        else:  # handle mini-batches later
            print("\t-> PROBLEM: mini-batches not supported yet. Choose batch_size 0 or -1")

        return error

    def _print_training_info(self, curr_epoch, total_epochs, curr_error, error_threshold):
        text = f"""{'-' * 45}\n\t-> training step: :{curr_epoch}/{total_epochs}\n\t\t* current error: {curr_error}, threshold: {error_threshold}\n"""
        print(text)

    def _train_helper(self, input_set, target_set, error_func):
        nnet_output = self.feedforward(input_set)
        error = error_func(target_set, nnet_output)

        self.backprop(target=target_set, output=nnet_output, error_func=error_func, )
        return error


#network hyperparameters

#-- topology
n_features = 2
n_hidden_1 = 4
n_outputs  = 1

topology   = [n_features, n_hidden_1, n_outputs]

#-- learning
learning_rate = 0.01
momentum = 0.1

nnet = NeuralNet(
    topology=topology,
    learning_rate=learning_rate,
    momentum=momentum,
    init_method='xavier',
    hidden_activation_func=relu,
)

X_train = np.array([[0.1, 0.3], [0.1, 0.4], [0.4, 0.9]])
y_train = np.array([[0.4], [0.5], [0.13]])

nnet.train(X_train, y_train, epochs=1000)


print(nnet.feedforward([0.1, 0.3]))
print(nnet.feedforward([0.1, 0.4]))
print(nnet.feedforward([0.4, 0.9]))

