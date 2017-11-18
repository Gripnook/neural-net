import unittest

from mock import MagicMock

from neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork((1, 2, 3))

    def test_train_stochastic(self):
        # given
        input_vectors = [1, 2, 3]
        output_vectors = [4, 5, 6]
        gradient_descent_method = 'stochastic'
        num_iterations = 100
        alpha = 0.5
        self.nn._stochastic_gradient_descent = MagicMock()

        # when
        self.nn.train(input_vectors, output_vectors, gradient_descent_method, num_iterations, alpha)

        # then
        self.nn._stochastic_gradient_descent.assert_called_with(input_vectors, output_vectors, num_iterations, alpha)

    def test_train_standard(self):
        # given
        input_vectors = [1, 2, 3]
        output_vectors = [4, 5, 6]
        gradient_descent_method = 'standard'
        num_iterations = 100
        alpha = 0.5
        self.nn._standard_gradient_descent = MagicMock()

        # when
        self.nn.train(input_vectors, output_vectors, gradient_descent_method, num_iterations, alpha)

        # then
        self.nn._standard_gradient_descent.assert_called_with(input_vectors, output_vectors, num_iterations, alpha)

    def test_train_invalid(self):
        # given
        input_vectors = [1, 2, 3]
        output_vectors = [4, 5, 6]
        gradient_descent_method = 'invalid'
        num_iterations = 100
        alpha = 0.5

        # then
        with self.assertRaises(ValueError):
            # when
            self.nn.train(input_vectors, output_vectors, gradient_descent_method, num_iterations, alpha)

    def test_predict(self):
        pass

    def test__standard_gradient_descent(self):
        pass

    def test__stochastic_gradient_descent(self):
        pass

    def test__get_gradient(self):
        pass

    def test__get_flattened_weights(self):
        pass

    def test__set_weights(self):
        pass


if __name__ == '__main__':
    unittest.main()
