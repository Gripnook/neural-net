import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        if self.num_layers <= 0:
            raise ValueError('Neural network must have at least one layer.')

        for layer_size in layer_sizes:
            if layer_size <= 0:
                raise ValueError('Neural network layers must have at least one neuron.')

        self._bias_weights = []
        self._weights = []
        for i in range(1, self.num_layers):
            self._bias_weights.append(np.random.uniform(-1, 1, (layer_sizes[i], 1)))
            self._weights.append(np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i - 1])))

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
        for w, W in zip(self._bias_weights, self._weights):
            propagated_input = sigmoid(w + W.dot(propagated_input))
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
        bias_gradients = [np.array([]) for _ in range(self.num_layers - 1)]
        gradients = [np.array([]) for _ in range(self.num_layers - 1)]
        outputs = [input_vector]

        # Forward propagation
        propagated_input = input_vector
        for w, W in zip(self._bias_weights, self._weights):
            propagated_input = sigmoid(w + W.dot(propagated_input))
            outputs.append(propagated_input)

        # Backward propagation
        delta = sigmoid_prime(outputs[self.num_layers - 1]).dot(expected_output_vector - outputs[self.num_layers - 1])
        bias_gradients[self.num_layers - 2] = - delta
        gradients[self.num_layers - 2] = - delta.dot(outputs[self.num_layers - 2].transpose())
        for i in range(self.num_layers - 3, -1, -1):
            delta = sigmoid_prime(outputs[i + 1]) * ((self._weights[i + 1].transpose()).dot(delta))
            bias_gradients[i] = - delta
            gradients[i] = - delta.dot(outputs[i].transpose())

        flattened_gradient = np.array([])
        for bias_gradient, gradient in zip(bias_gradients, gradients):
            flattened_gradient = np.append(flattened_gradient, bias_gradient)
            flattened_gradient = np.append(flattened_gradient, gradient)
        return flattened_gradient

    def _get_flattened_weights(self):
        """
        Flattens the weight matrices to produce a single vector.

        :return: the weights as a flattened matrix (vector)
        """
        flattened_weights = np.array([])
        for w, W in zip(self._bias_weights, self._weights):
            flattened_weights = np.append(flattened_weights, w)
            flattened_weights = np.append(flattened_weights, W)
        return flattened_weights

    def _set_weights(self, flattened_weights):
        """
        Updates the weight matrices based on a flattened vector.

        :param flattened_weights: the flattened weight vector
        """
        index = 0
        for w, W in zip(self._bias_weights, self._weights):
            w[:] = np.reshape(flattened_weights[index:index + w.size], w.shape)
            index += w.size
            W[:] = np.reshape(flattened_weights[index:index + W.size], W.shape)
            index += W.size


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
