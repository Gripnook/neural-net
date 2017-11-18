import numpy as np

from scipy import optimize
from copy import deepcopy


class Trainer(object):
    def __init__(self, nn):
        self._nn = nn

    def train(self, input_vectors, output_vectors, method='stochastic',
              num_iterations=1000, alpha=0.1):
        if method is 'stochastic':
            self._stochastic_gradient_descent(input_vectors, output_vectors,
                                              num_iterations, alpha)
        elif method is 'standard':
            self._standard_gradient_descent(input_vectors, output_vectors,
                                            num_iterations, alpha)
        elif method is 'BFGS':
            self._bfgs(input_vectors, output_vectors)
        else:
            raise ValueError('Invalid gradient descent method')

    def _stochastic_gradient_descent(self, input_vectors, output_vectors,
                                     num_iterations, alpha):
        for _ in range(num_iterations):
            for input_vector, output_vector in zip(input_vectors, output_vectors):
                weights = self._nn.get_weights()
                gradients = self._nn.get_gradient(input_vector, output_vector)
                for weight, gradient in zip(weights, gradients):
                    weight -= alpha * gradient
                self._nn.set_weights(weights)

    def _standard_gradient_descent(self, input_vectors, output_vectors,
                                   num_iterations, alpha):
        for _ in range(num_iterations):
            weights = self._nn.get_weights()
            total_gradients = [np.zeros(weight.shape) for weight in weights]
            for input_vector, output_vector in zip(input_vectors, output_vectors):
                gradients = self._nn.get_gradient(input_vector, output_vector)
                for gradient, total_gradient in zip(gradients, total_gradients):
                    total_gradient += gradient
            for weight, total_gradient in zip(weights, total_gradients):
                weight -= alpha * total_gradient
            self._nn.set_weights(weights)

    def _bfgs(self, input_vectors, output_vectors):
        def flatten(weights):
            flattened_weights = np.array([])
            for weight in weights:
                flattened_weights = np.append(flattened_weights, weight)
            return flattened_weights

        def inflate(flattened_weights):
            index = 0
            weights = deepcopy(self._nn.get_weights())
            for weight in weights:
                weight[:] = np.reshape(flattened_weights[index:index + weight.size], weight.shape)
                index += weight.size
            return weights

        def loss(weights):
            self._nn.set_weights(inflate(weights))
            gradient = np.zeros(weights.shape)
            for input_vector, output_vector in zip(input_vectors, output_vectors):
                gradient += flatten(self._nn.get_gradient(input_vector, output_vector))
            return self._nn.get_loss(np.hstack(input_vectors), np.hstack(output_vectors)), gradient

        optimize.minimize(loss, flatten(self._nn.get_weights()), method='BFGS', jac=True)
