import unittest

import numpy as np

from copy import deepcopy

from neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork((2, 5, 1))
        self.weights = deepcopy(self.nn.get_weights())
        self.input_vector = np.array([[1, 1]])
        self.output_vector = np.array([[2]])

    def test_predict(self):
        pass

    def test_get_weights(self):
        pass

    def test_set_weights(self):
        pass

    def test_get_gradient(self):
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self._test_get_gradient(i, j, k)

    def _test_get_gradient(self, i, j, k):
        self.nn.set_weights(deepcopy(self.weights))
        gradient = self.nn.get_gradient(self.input_vector, self.output_vector)
        approx_gradient = self._get_approx_gradient(i, j, k)
        self.assertAlmostEqual(gradient[i][j, k], approx_gradient)

    def _get_approx_gradient(self, i, j, k):
        epsilon = 0.001
        weights = deepcopy(self.weights)
        weights[i][j, k] += epsilon
        self.nn.set_weights(weights)
        loss1 = self.nn.get_loss([self.input_vector], [self.output_vector])
        weights = deepcopy(self.weights)
        weights[i][j, k] -= epsilon
        self.nn.set_weights(weights)
        loss2 = self.nn.get_loss([self.input_vector], [self.output_vector])
        return (loss1 - loss2) / (2 * epsilon)


if __name__ == '__main__':
    unittest.main()
