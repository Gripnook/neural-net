import logging
import random

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        if self.num_layers <= 0:
            raise ValueError('Neural network must have at least one layer.')

        for layer_size in layer_sizes:
            if layer_size <= 0:
                raise ValueError('Neural network layers must have at least one neuron.')

        self._weights = []
        for i in range(1, self.num_layers):
            self._weights.append(np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i - 1] + 1)))

    def train(self, input_vectors, output_vectors, gradient_descent_method='stochastic',
              num_iterations=1000, alpha=0.1):
        if gradient_descent_method is 'stochastic':
            self._stochastic_gradient_descent(input_vectors, output_vectors, num_iterations, alpha)
        elif gradient_descent_method is 'standard':
            self._standard_gradient_descent(input_vectors, output_vectors, num_iterations, alpha)
        else:
            raise ValueError('Invalid gradient descent method.')

    def predict(self, input_vectors):
        """
        Predicts the output for the given input data.

        :param input_vectors: the input data
        :return: the predicted output
        """
        propagated_input = np.hstack(input_vectors)
        for w in self._weights:
            propagated_input = sigmoid(w.dot(expand(propagated_input)))
        return propagated_input

    def _standard_gradient_descent(self, input_vectors, output_vectors, num_iterations, alpha):
        """
        Performs standard gradient descent.

        :param input_vectors: the input vectors
        :param output_vectors: the output vectors
        :param num_iterations: the number of training iterations
        :param alpha: the learning rate
        """
        for _ in range(num_iterations):
            gradient_sum = sum([self._get_gradient(in_example, out_example)
                                for in_example, out_example in zip(input_vectors, output_vectors)])
            self._set_weights(self._get_flattened_weights() - alpha * gradient_sum)

    def _stochastic_gradient_descent(self, input_vectors, output_vectors, num_iterations, alpha):
        """
        Performs stochastic gradient descent.

        :param input_vectors: the input vectors
        :param output_vectors: the output vectors
        :param num_iterations: the number of training iterations
        :param alpha: the learning rate
        """
        for _ in range(num_iterations):
            for in_example, out_example in zip(input_vectors, output_vectors):
                self._set_weights(self._get_flattened_weights() - alpha * self._get_gradient(in_example, out_example))

    def _get_gradient(self, input_vector, expected_output_vector):
        """
        Computes the gradient of the L2 loss with respect to the weight vector for the given training example input
        and output.

        :param input_vector: the input data
        :param expected_output_vector: the expected output data
        :return: the gradient of the loss
        """
        gradients = [np.array([]) for _ in range(self.num_layers - 1)]
        outputs = [input_vector]

        # Forward propagation
        propagated_input = input_vector
        for w in self._weights:
            propagated_input = sigmoid(w.dot(expand(propagated_input)))
            outputs.append(propagated_input)

        # Backward propagation
        delta = sigmoid_prime(outputs[self.num_layers - 1]).dot(expected_output_vector - outputs[self.num_layers - 1])
        gradients[self.num_layers - 2] = - delta.dot(expand(outputs[self.num_layers - 2]).transpose())
        for i in range(self.num_layers - 3, -1, -1):
            delta = sigmoid_prime(outputs[i + 1]) * ((reduce_weights(self._weights[i + 1]).transpose()).dot(delta))
            gradients[i] = - delta.dot(expand(outputs[i]).transpose())

        flattened_gradient = np.array([])
        for gradient in gradients:
            flattened_gradient = np.append(flattened_gradient, gradient)
        return flattened_gradient

    def _get_flattened_weights(self):
        """
        Flattens the weight matrices to produce a single vector.

        :return: the weights as a flattened matrix (vector)
        """
        flattened_weights = np.array([])
        for w in self._weights:
            flattened_weights = np.append(flattened_weights, w)
        return flattened_weights

    def _set_weights(self, flattened_weights):
        """
        Updates the weight matrices based on a flattened vector.

        :param flattened_weights: the flattened weight vector
        """
        index = 0
        for w in self._weights:
            for i in range(w.size):
                weight_index = np.unravel_index([i], w.shape)
                w[weight_index] = flattened_weights[index]
                index += 1


def expand(output_vector):
    """
    Expands the given output_vector to include a dummy input to represent the bias.

    :param output_vector: the output vector
    :return: the expanded output vector
    """
    num_cols = output_vector.shape[1] if len(output_vector.shape) > 1 else output_vector.shape
    bias = np.ones(num_cols)
    return np.vstack((bias, output_vector))


def reduce_weights(expanded_weights):
    """
    Reduces the given weight matrix to exclude the dummy input weight.

    :param expanded_weights: the expanded weights
    :return: the reduced weights
    """
    return expanded_weights[:, 1:]


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

    :param y: the y value (sigmoid value)
    :return: the derivative value
    """
    return y * (1 - y)
