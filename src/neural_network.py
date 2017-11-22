import numpy as np


class NeuralNetwork(object):
    def __init__(self, layer_sizes, sigmoid='tanh', max_initial_weight=0.05):
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

    def reset_weights(self, max_initial_weight=0.05):
        """
        Resets the weights to small values.
        """
        for i in range(1, self._num_layers):
            self._bias_weights[i - 1] = np.random.uniform(-max_initial_weight, max_initial_weight, (self._layer_sizes[i], 1))
            self._weights[i - 1] = np.random.uniform(-max_initial_weight, max_initial_weight, (self._layer_sizes[i], self._layer_sizes[i - 1]))

    def predict(self, input_vectors):
        """
        Predicts the output for the given input data.

        :param input_vectors: the input data
        :return: the predicted output
        """
        propagated_input = np.vstack(input_vectors).transpose()
        for bias_weight, weight in zip(self._bias_weights, self._weights):
            propagated_input = self._sigmoid(bias_weight + weight.dot(propagated_input))
        return np.expand_dims(propagated_input.transpose(), 1)

    def get_weights(self):
        """
        Returns the weights as a list of matrices.

        :return: the weights as a list of matrices
        """
        return self._bias_weights + self._weights

    def set_weights(self, weights):
        """
        Updates the weight matrices with the given list of matrices.

        :param weights: the weights as a list of matrices
        """
        self._bias_weights = weights[:self._num_layers - 1]
        self._weights = weights[self._num_layers - 1:]

    def get_loss(self, input_vectors, expected_output_vectors):
        """
        Computes the L2 loss for the given training examples.

        :param input_vectors: the input data
        :param expected_output_vectors: the expected output data
        :return: the L2 loss
        """
        return 0.5 * np.sum((np.array(expected_output_vectors) - self.predict(input_vectors)) ** 2)

    def get_gradient(self, input_vector, expected_output_vector):
        """
        Computes the gradient of the L2 loss with respect to the weights for the given training example.

        :param input_vector: the input data
        :param expected_output_vector: the expected output data
        :return: the gradient of the L2 loss as a list of matrices
        """
        bias_gradients = [np.array([]) for _ in range(self._num_layers - 1)]
        gradients = [np.array([]) for _ in range(self._num_layers - 1)]
        outputs = [input_vector.transpose()]

        # Forward propagation.
        propagated_input = outputs[0]
        for bias_weight, weight in zip(self._bias_weights, self._weights):
            propagated_input = self._sigmoid(bias_weight + weight.dot(propagated_input))
            outputs.append(propagated_input)

        # Backward propagation.
        delta = self._sigmoid_prime(outputs[self._num_layers - 1]) * (
            expected_output_vector.transpose() - outputs[self._num_layers - 1])
        bias_gradients[self._num_layers - 2] = -delta
        gradients[self._num_layers - 2] = -delta.dot(outputs[self._num_layers - 2].transpose())
        for i in range(self._num_layers - 3, -1, -1):
            delta = self._sigmoid_prime(outputs[i + 1]) * ((self._weights[i + 1].transpose()).dot(delta))
            bias_gradients[i] = -delta
            gradients[i] = -delta.dot(outputs[i].transpose())

        return bias_gradients + gradients


def logistic_sigmoid(x):
    """
    Computes the sigmoid function at the given x value.

    :param x: the x value
    :return: the sigmoid function at the given x value
    """
    return 1 / (1 + np.exp(-x))


def logistic_sigmoid_prime(y):
    """
    Computes the derivative of the sigmoid function at the given value of y = f(x).

    :param y: the value of the sigmoid function
    :return: the derivative at the given y value
    """
    return y * (1 - y)


def tanh_sigmoid(x):
    """
    Computes the tanh sigmoid function at the given x value.

    :param x: the x value
    :return: the tanh sigmoid function at the given x value
    """
    return 1.7159 * np.tanh(2.0 * x / 3.0)


def tanh_sigmoid_prime(y):
    """
    Computes the derivative of the tanh sigmoid function at the given value of y = f(x).

    :param y: the value of the tanh sigmoid function
    :return: the derivative at the given y value
    """
    return 2.0 * (1.7159 - (y ** 2) / 1.7159) / 3.0
