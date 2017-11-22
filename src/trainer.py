from __future__ import division

import logging
import random
import numpy as np

from scipy import optimize
from copy import deepcopy


class Trainer(object):
    def __init__(self, nn):
        self._nn = nn

    def train(self, input_vectors, output_vectors, method='stochastic',
              num_iterations=1000, learning_rate=0.1, momentum=0.1, callback=lambda iteration: None,
              batch_size=100):
        if method is 'stochastic':
            self._stochastic_gradient_descent(input_vectors, output_vectors,
                                              num_iterations, learning_rate, momentum, callback,
                                              batch_size)
        elif method is 'standard':
            self._standard_gradient_descent(input_vectors, output_vectors,
                                            num_iterations, learning_rate)
        elif method is 'BFGS':
            self._bfgs(input_vectors, output_vectors)
        else:
            raise ValueError('Invalid gradient descent method')

    def k_fold_cross_validation(self, input_vectors, output_vectors, k=10, method='stochastic', num_iterations=1000,
                                learning_rate=0.1):
        if k > len(input_vectors) or k > len(output_vectors):
            raise ValueError('k cannot be greater than the total number of input/output vector pairs.')

        if len(input_vectors) != len(output_vectors):
            raise ValueError('Must have same number of input and output vectors.')

        losses = []
        index_offset = int(round(len(input_vectors) / k))
        for i in range(k):
            # Split training/testing data
            test_start_index = i * index_offset
            test_end_index = (i + 1) * index_offset if i < k - 1 else len(input_vectors)
            train_input = np.vstack((input_vectors[:test_start_index], input_vectors[test_end_index:]))
            train_output = np.vstack((output_vectors[:test_start_index], output_vectors[test_end_index:]))
            test_input = input_vectors[test_start_index:test_end_index]
            test_output = output_vectors[test_start_index:test_end_index]

            # Train
            self._nn.reset_weights()
            self.train(train_input, train_output, method, num_iterations, learning_rate)

            # Test
            loss = self._nn.get_loss(test_input, test_output)
            losses.append(loss)
        return np.mean(losses)

    def _stochastic_gradient_descent(self, input_vectors, output_vectors,
                                     num_iterations, learning_rate, momentum,
                                     callback, batch_size):
        delta_weights = []
        for weight in self._nn.get_weights():
            delta_weights.append(np.zeros(weight.shape))
        learning_rate /= 100
        for iteration in range(num_iterations):
            if iteration % 10000 == 0:
                learning_rate *= 0.99
            idx = np.random.randint(input_vectors.shape[0], size=100)
            input_vector = input_vectors[idx,:,:]
            output_vector = output_vectors[idx,:,:]
            weights = self._nn.get_weights()
            gradients = self._nn.get_gradient(input_vector, output_vector)
            for delta_weight, weight, gradient in zip(delta_weights, weights, gradients):
                delta_weight[:] = -learning_rate * gradient + momentum * delta_weight
                weight += delta_weight
            self._nn.set_weights(weights)
            callback(iteration)

    def _standard_gradient_descent(self, input_vectors, output_vectors,
                                   num_iterations, learning_rate):
        for _ in range(num_iterations):
            weights = self._nn.get_weights()
            gradients = self._nn.get_gradient(input_vectors, output_vectors)
            for weight, gradient in zip(weights, gradients):
                weight -= learning_rate * gradient
            self._nn.set_weights(weights)

    # def _bfgs(self, input_vectors, output_vectors):
    #     def flatten(weights):
    #         flattened_weights = np.array([])
    #         for weight in weights:
    #             flattened_weights = np.append(flattened_weights, weight)
    #         return flattened_weights

    #     def inflate(flattened_weights):
    #         index = 0
    #         weights = deepcopy(self._nn.get_weights())
    #         for weight in weights:
    #             weight[:] = np.reshape(flattened_weights[index:index + weight.size], weight.shape)
    #             index += weight.size
    #         return weights

    #     def loss(weights):
    #         self._nn.set_weights(inflate(weights))
    #         gradient = np.zeros(weights.shape)
    #         for input_vector, output_vector in zip(input_vectors, output_vectors):
    #             gradient += flatten(self._nn.get_gradient(input_vector, output_vector))
    #         return self._nn.get_loss(input_vectors, output_vectors), gradient

    #     optimize.minimize(loss, flatten(self._nn.get_weights()), method='BFGS', jac=True)
