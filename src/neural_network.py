import numpy as np


class NeuralNetwork(object):
    def __init__(self, layer_sizes):
        self._num_layers = len(layer_sizes)
        if self._num_layers <= 0:
            raise ValueError('Neural network must have at least one layer.')

        for layer_size in layer_sizes:
            if layer_size <= 0:
                raise ValueError('Neural network layers must have at least one neuron.')

        self._bias_weights = []
        self._weights = []
        for i in range(1, self._num_layers):
            self._bias_weights.append(np.random.uniform(-1, 1, (layer_sizes[i], 1)))
            self._weights.append(np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i - 1])))

    def predict(self, input_vectors):
        """
        Predicts the output for the given input data.

        :param input_vectors: the input data
        :return: the predicted output
        """
        propagated_input = np.hstack(input_vectors)
        for bias_weight, weight in zip(self._bias_weights, self._weights):
            propagated_input = sigmoid(bias_weight + weight.dot(propagated_input))
        return propagated_input

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
        return 0.5 * np.sum((np.hstack(expected_output_vectors) - self.predict(input_vectors)) ** 2)

    def get_gradient(self, input_vector, expected_output_vector):
        """
        Computes the gradient of the L2 loss with respect to the weights for the given training example.

        :param input_vector: the input data
        :param expected_output_vector: the expected output data
        :return: the gradient of the L2 loss as a list of matrices
        """
        bias_gradients = [np.array([]) for _ in range(self._num_layers - 1)]
        gradients = [np.array([]) for _ in range(self._num_layers - 1)]
        outputs = [input_vector]

        # Forward propagation.
        propagated_input = input_vector
        for bias_weight, weight in zip(self._bias_weights, self._weights):
            propagated_input = sigmoid(bias_weight + weight.dot(propagated_input))
            outputs.append(propagated_input)

        # Backward propagation.
        delta = sigmoid_prime(outputs[self._num_layers - 1]) * (expected_output_vector - outputs[self._num_layers - 1])
        bias_gradients[self._num_layers - 2] = -delta
        gradients[self._num_layers - 2] = -delta.dot(outputs[self._num_layers - 2].transpose())
        for i in range(self._num_layers - 3, -1, -1):
            delta = sigmoid_prime(outputs[i + 1]) * ((self._weights[i + 1].transpose()).dot(delta))
            bias_gradients[i] = -delta
            gradients[i] = -delta.dot(outputs[i].transpose())

        return bias_gradients + gradients


def sigmoid(x):
    """
    Computes the sigmoid function at the given x value.

    :param x: the x value
    :return: the sigmoid function at the given x value
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(y):
    """
    Computes the derivative of the sigmoid function at the given value of y = f(x).

    :param y: the value of the sigmoid function
    :return: the derivative at the given y value
    """
    return y * (1 - y)
