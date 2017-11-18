import unittest

import numpy as np

from mock import MagicMock

from trainer import Trainer
from neural_network import NeuralNetwork


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork((2, 5, 1))
        self.trainer = Trainer(self.nn)

    def test_train_stochastic(self):
        # given
        input_vectors = [np.array([1, 1]).transpose(),
                         np.array([2, 3]).transpose(),
                         np.array([2, 1]).transpose()]
        output_vectors = [np.array([2]), np.array([13]), np.array([5])]
        method = 'stochastic'
        num_iterations = 100
        alpha = 0.1
        self.trainer._stochastic_gradient_descent = MagicMock()

        # when
        self.trainer.train(input_vectors, output_vectors, method, num_iterations, alpha)

        # then
        self.trainer._stochastic_gradient_descent.assert_called_with(input_vectors, output_vectors,
                                                                     num_iterations, alpha)

    def test_train_standard(self):
        # given
        input_vectors = [np.array([1, 1]).transpose(),
                         np.array([2, 3]).transpose(),
                         np.array([2, 1]).transpose()]
        output_vectors = [np.array([2]), np.array([13]), np.array([5])]
        method = 'standard'
        num_iterations = 100
        alpha = 0.1
        self.trainer._standard_gradient_descent = MagicMock()

        # when
        self.trainer.train(input_vectors, output_vectors, method, num_iterations, alpha)

        # then
        self.trainer._standard_gradient_descent.assert_called_with(input_vectors, output_vectors,
                                                                   num_iterations, alpha)

    def test_train_invalid(self):
        # given
        input_vectors = [np.array([1, 1]).transpose(),
                         np.array([2, 3]).transpose(),
                         np.array([2, 1]).transpose()]
        output_vectors = [np.array([2]), np.array([13]), np.array([5])]
        method = 'invalid'
        num_iterations = 100
        alpha = 0.1

        # then
        with self.assertRaises(ValueError):
            # when
            self.trainer.train(input_vectors, output_vectors, method, num_iterations, alpha)


if __name__ == '__main__':
    unittest.main()
