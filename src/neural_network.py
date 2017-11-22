import numpy as np


class NeuralNetwork(object):
    def __init__(self, layer_sizes, max_initial_weight=0.05, sigmoid='tanh'):
        self._layer_sizes = layer_sizes
        self._num_layers = len(self._layer_sizes)
        if self._num_layers <= 0:
            raise ValueError('Neural network must have at least one layer.')

        for layer_size in self._layer_sizes:
            if layer_size <= 0:
                raise ValueError('Neural network layers must have at least one neuron.')

        self._bias_weights = [np.array([]) for _ in range(self._num_layers - 1)]
        self._weights = [np.array([]) for _ in range(self._num_layers - 1)]
        self.reset_weights(max_initial_weight)

        if sigmoid is 'logistic':
            self._sigmoid = logistic_sigmoid
            self._sigmoid_prime = logistic_sigmoid_prime
        elif sigmoid is 'tanh':
            self._sigmoid = tanh_sigmoid
            self._sigmoid_prime = tanh_sigmoid_prime
        else:
            raise ValueError('Invalid sigmoid function.')

    def predict(self, input_vectors):
        """
        Predicts the output vectors for the given input vectors.
        The input must consist of a 2D or 3D numpy array containing one vector per row.
        """

        # Stack the input vectors into a 2D array for propagation.
        prediction = np.vstack(input_vectors).transpose()

        # Propagate the input through the network.
        for bias_weight, weight in zip(self._bias_weights, self._weights):
            prediction = self._sigmoid(bias_weight + weight.dot(prediction))

        if len(input_vectors.shape) == 3:
            # Undo the stacking to output a 3D array.
            return np.expand_dims(prediction.transpose(), 1)
        else:
            return prediction.transpose()

    def get_weights(self):
        """
        Returns the network weights as a list of numpy arrays.
        """

        return self._bias_weights + self._weights

    def set_weights(self, weights):
        """
        Sets the network weights to the given list of numpy arrays.
        """

        self._bias_weights = weights[:self._num_layers - 1]
        self._weights = weights[self._num_layers - 1:]

    def reset_weights(self, max_initial_weight=0.05):
        """
        Resets the network weights to random values.
        """

        for i in range(1, self._num_layers):
            self._bias_weights[i - 1] = np.random.uniform(-max_initial_weight, max_initial_weight,
                                                          (self._layer_sizes[i], 1))
            self._weights[i - 1] = np.random.uniform(-max_initial_weight, max_initial_weight,
                                                     (self._layer_sizes[i], self._layer_sizes[i - 1]))

    def get_loss(self, input_vectors, expected_output_vectors):
        """
        Computes the L2 loss for the given training examples.
        """

        return 0.5 * np.sum((expected_output_vectors - self.predict(input_vectors)) ** 2)

    def get_loss_gradient(self, input_vectors, expected_output_vectors):
        """
        Computes the gradient of the L2 loss with respect to the network weights for
        the given training examples. Returns the result as a list of numpy arrays.
        """

        bias_gradients = [np.array([]) for _ in range(self._num_layers - 1)]
        gradients = [np.array([]) for _ in range(self._num_layers - 1)]

        # Stack the input vectors into a 2D array for propagation.
        outputs = [np.vstack(input_vectors).transpose()]

        # Propagate the input through the network.
        prediction = outputs[0]
        for bias_weight, weight in zip(self._bias_weights, self._weights):
            prediction = self._sigmoid(bias_weight + weight.dot(prediction))
            outputs.append(prediction)

        # Propagate the error backwards through the network.
        delta = self._sigmoid_prime(outputs[self._num_layers - 1]) * (
            np.vstack(expected_output_vectors).transpose() - outputs[self._num_layers - 1])
        bias_gradients[self._num_layers - 2] = -np.sum(delta, 1).reshape(-1, 1)
        gradients[self._num_layers - 2] = -delta.dot(outputs[self._num_layers - 2].transpose())
        for i in range(self._num_layers - 3, -1, -1):
            delta = self._sigmoid_prime(outputs[i + 1]) * ((self._weights[i + 1].transpose()).dot(delta))
            bias_gradients[i] = -np.sum(delta, 1).reshape(-1, 1)
            gradients[i] = -delta.dot(outputs[i].transpose())
        return bias_gradients + gradients


def logistic_sigmoid(x):
    """
    Computes the logistic sigmoid function at the given x value.
    """

    return 1 / (1 + np.exp(-x))


def logistic_sigmoid_prime(y):
    """
    Computes the derivative of the logistic sigmoid function at the given value of y = f(x).
    """

    return y * (1 - y)


def tanh_sigmoid(x):
    """
    Computes the tanh sigmoid function at the given x value.
    """

    return 1.7159 * np.tanh(2.0 * x / 3.0)


def tanh_sigmoid_prime(y):
    """
    Computes the derivative of the tanh sigmoid function at the given value of y = f(x).
    """

    return 2.0 * (1.7159 - (y ** 2) / 1.7159) / 3.0
