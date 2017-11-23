import unittest

import numpy as np
from copy import deepcopy

from neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def test_nn_with_single_layer_predicts_the_input_for_1D_array(self):
        nn = NeuralNetwork((3,))
        input_vector = np.array([[1, 1, 2]])
        np.testing.assert_array_equal(nn.predict(input_vector), input_vector)

    def test_nn_with_single_layer_predicts_the_input_for_2D_array(self):
        nn = NeuralNetwork((3,))
        input_vectors = np.array([[1, 1, 2], [2, 0, 1]])
        np.testing.assert_array_equal(nn.predict(input_vectors), input_vectors)

    def test_nn_with_single_layer_predicts_the_input_for_3D_array(self):
        nn = NeuralNetwork((3,))
        input_vectors = np.array([[[1, 1, 2]], [[2, 0, 1]]])
        np.testing.assert_array_equal(nn.predict(input_vectors), input_vectors)

    def test_nn_with_logistic_sigmoid_has_correct_loss_for_single_example(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='logistic')
        input_vector = np.array([[1, 1]])
        output_vector = np.array([[2]])
        self._test_loss(nn, input_vector, output_vector)

    def test_nn_with_logistic_sigmoid_has_correct_loss_for_multiple_examples(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='logistic')
        input_vectors = np.array([[[1, 1]], [[-1, 2]]])
        output_vectors = np.array([[[2]], [[5]]])
        self._test_loss(nn, input_vectors, output_vectors)

    def test_nn_with_tanh_sigmoid_has_correct_loss_for_single_example(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='tanh')
        input_vector = np.array([[1, 1]])
        output_vector = np.array([[2]])
        self._test_loss(nn, input_vector, output_vector)

    def test_nn_with_tanh_sigmoid_has_correct_loss_for_multiple_examples(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='tanh')
        input_vectors = np.array([[[1, 1]], [[-1, 2]]])
        output_vectors = np.array([[[2]], [[5]]])
        self._test_loss(nn, input_vectors, output_vectors)

    def test_nn_with_logistic_sigmoid_has_correct_loss_gradient_for_single_example(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='logistic')
        input_vector = np.array([[1, 1]])
        output_vector = np.array([[2]])
        self._test_gradient(nn, input_vector, output_vector)

    def test_nn_with_logistic_sigmoid_has_correct_loss_gradient_for_multiple_examples(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='logistic')
        input_vectors = np.array([[[1, 1]], [[-1, 2]]])
        output_vectors = np.array([[[2]], [[5]]])
        self._test_gradient(nn, input_vectors, output_vectors)

    def test_nn_with_tanh_sigmoid_has_correct_loss_gradient_for_single_example(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='tanh')
        input_vector = np.array([[1, 1]])
        output_vector = np.array([[2]])
        self._test_gradient(nn, input_vector, output_vector)

    def test_nn_with_tanh_sigmoid_has_correct_loss_gradient_for_multiple_examples(self):
        nn = NeuralNetwork((2, 5, 1), sigmoid='tanh')
        input_vectors = np.array([[[1, 1]], [[-1, 2]]])
        output_vectors = np.array([[[2]], [[5]]])
        self._test_gradient(nn, input_vectors, output_vectors)

    def test_nn_with_invalid_sigmoid_throws_error(self):
        with self.assertRaises(ValueError):
            nn = NeuralNetwork((2, 5, 1), sigmoid='invalid')

    def _test_loss(self, nn, input_vectors, output_vectors):
        prediction = nn.predict(input_vectors)
        self.assertAlmostEqual(nn.get_loss(input_vectors, output_vectors),
                               0.5 * sum((output_vectors - prediction) ** 2))

    def _test_gradient(self, nn, input_vectors, output_vectors):
        weights = nn.get_weights()
        for layer in range(len(weights)):
            for row in range(weights[layer].shape[0]):
                for col in range(weights[layer].shape[1]):
                    gradient = nn.get_loss_gradient(input_vectors, output_vectors)
                    approx_gradient = self._get_approx_gradient(nn, input_vectors, output_vectors, layer, row, col)
                    self.assertAlmostEqual(gradient[layer][row, col], approx_gradient)

    def _get_approx_gradient(self, nn, input_vectors, output_vectors, layer, row, col, epsilon=1e-6):
        saved_weights = nn.get_weights()
        weights = deepcopy(saved_weights)
        weights[layer][row, col] += epsilon
        nn.set_weights(weights)
        loss1 = nn.get_loss(input_vectors, output_vectors)
        weights = deepcopy(saved_weights)
        weights[layer][row, col] -= epsilon
        nn.set_weights(weights)
        loss2 = nn.get_loss(input_vectors, output_vectors)
        nn.set_weights(saved_weights)
        return (loss1 - loss2) / (2 * epsilon)


if __name__ == '__main__':
    unittest.main()
