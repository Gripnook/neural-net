import unittest

import numpy as np

from mock import MagicMock

from trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.nn = MagicMock()
        self.trainer = Trainer(self.nn)

    def test_train_stochastic(self):
        # given
        input_vectors = np.array([
            [[1, 1]], [[2, 3]], [[2, 1]]
        ])
        output_vectors = np.array([
            [[2]], [[13]], [[5]]
        ])
        method = 'stochastic'
        num_iterations = 100
        alpha = 0.1
        momentum = 0.1
        self.trainer._stochastic_gradient_descent = MagicMock()

        # when
        self.trainer.train(input_vectors, output_vectors, method, num_iterations, alpha, momentum)

        # then
        self.trainer._stochastic_gradient_descent.assert_called_with(input_vectors, output_vectors,
                                                                     num_iterations, alpha, momentum)

    def test_train_standard(self):
        # given
        input_vectors = np.array([
            [[1, 1]], [[2, 3]], [[2, 1]]
        ])
        output_vectors = np.array([
            [[2]], [[13]], [[5]]
        ])
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
        input_vectors = np.array([
            [[1, 1]], [[2, 3]], [[2, 1]]
        ])
        output_vectors = np.array([
            [[2]], [[13]], [[5]]
        ])
        method = 'invalid'
        num_iterations = 100
        alpha = 0.1

        # then
        with self.assertRaises(ValueError):
            # when
            self.trainer.train(input_vectors, output_vectors, method, num_iterations, alpha)

    def test_k_fold_cross_validation_even_split(self):
        # given
        input_vectors = np.array([
            [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]]
        ])
        input_split_1 = np.array([
            [[1, 1]], [[2, 2]]
        ])
        input_split_2 = np.array([
            [[3, 3]], [[4, 4]]
        ])
        output_vectors = np.array([
            [[1]], [[2]], [[3]], [[4]]
        ])
        output_split_1 = np.array([
            [[1]], [[2]]
        ])
        output_split_2 = np.array([
            [[3]], [[4]]
        ])
        method = 'method'
        num_iterations = 100
        alpha = 0.1
        k = 2

        # given
        self.trainer.train = MagicMock()
        self.trainer._nn.get_loss.return_value = 1
        expected_mean = 1

        # when
        actual_mean = self.trainer.k_fold_cross_validation(input_vectors, output_vectors, k, method, num_iterations,
                                                           alpha)

        # then
        self.assertEqual(expected_mean, actual_mean)

        self.trainer._nn.reset_weights.assert_called()

        train_calls = self.trainer.train.call_args_list
        np.testing.assert_array_equal(input_split_1, train_calls[1][0][0])
        np.testing.assert_array_equal(input_split_2, train_calls[0][0][0])
        np.testing.assert_array_equal(output_split_1, train_calls[1][0][1])
        np.testing.assert_array_equal(output_split_2, train_calls[0][0][1])

        get_loss_calls = self.trainer._nn.get_loss.call_args_list
        np.testing.assert_array_equal(input_split_1, get_loss_calls[0][0][0])
        np.testing.assert_array_equal(input_split_2, get_loss_calls[1][0][0])
        np.testing.assert_array_equal(output_split_1, get_loss_calls[0][0][1])
        np.testing.assert_array_equal(output_split_2, get_loss_calls[1][0][1])

    def test_k_fold_cross_validation_uneven_split(self):
        # given
        input_vectors = np.array([
            [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]], [[5, 5]]
        ])
        input_split_1 = np.array([
            [[1, 1]], [[2, 2]], [[3, 3]]
        ])
        input_split_2 = np.array([
            [[4, 4]], [[5, 5]]
        ])
        output_vectors = np.array([
            [[1]], [[2]], [[3]], [[4]], [[5]]
        ])
        output_split_1 = np.array([
            [[1]], [[2]], [[3]]
        ])
        output_split_2 = np.array([
            [[4]], [[5]]
        ])
        method = 'method'
        num_iterations = 100
        alpha = 0.1
        k = 2

        # given
        self.trainer.train = MagicMock()
        self.trainer._nn.get_loss.return_value = 1
        expected_mean = 1

        # when
        actual_mean = self.trainer.k_fold_cross_validation(input_vectors, output_vectors, k, method, num_iterations,
                                                           alpha)

        # then
        self.assertEqual(expected_mean, actual_mean)

        self.trainer._nn.reset_weights.assert_called()

        train_calls = self.trainer.train.call_args_list
        np.testing.assert_array_equal(input_split_1, train_calls[1][0][0])
        np.testing.assert_array_equal(input_split_2, train_calls[0][0][0])
        np.testing.assert_array_equal(output_split_1, train_calls[1][0][1])
        np.testing.assert_array_equal(output_split_2, train_calls[0][0][1])

        get_loss_calls = self.trainer._nn.get_loss.call_args_list
        np.testing.assert_array_equal(input_split_1, get_loss_calls[0][0][0])
        np.testing.assert_array_equal(input_split_2, get_loss_calls[1][0][0])
        np.testing.assert_array_equal(output_split_1, get_loss_calls[0][0][1])
        np.testing.assert_array_equal(output_split_2, get_loss_calls[1][0][1])

    def test_k_fold_cross_validation_stacked(self):
        # given
        input_vectors = np.array([
            [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]], [[5, 5]]
        ])
        input_split_1 = np.array([
            [[1, 1]], [[2, 2]]
        ])
        input_split_2 = np.array([
            [[3, 3]], [[4, 4]]
        ])
        input_split_3 = np.array([
            [[5, 5]]
        ])
        output_vectors = np.array([
            [[1]], [[2]], [[3]], [[4]], [[5]]
        ])
        output_split_1 = np.array([
            [[1]], [[2]]
        ])
        output_split_2 = np.array([
            [[3]], [[4]]
        ])
        output_split_3 = np.array([
            [[5]]
        ])
        method = 'method'
        num_iterations = 100
        alpha = 0.1
        k = 3

        # given
        self.trainer.train = MagicMock()
        self.trainer._nn.get_loss.return_value = 1
        expected_mean = 1

        # when
        actual_mean = self.trainer.k_fold_cross_validation(input_vectors, output_vectors, k, method, num_iterations,
                                                           alpha)

        # then
        self.assertEqual(expected_mean, actual_mean)

        self.trainer._nn.reset_weights.assert_called()

        train_calls = self.trainer.train.call_args_list
        np.testing.assert_array_equal(np.vstack((input_split_2, input_split_3)), train_calls[0][0][0])
        np.testing.assert_array_equal(np.vstack((input_split_1, input_split_3)), train_calls[1][0][0])
        np.testing.assert_array_equal(np.vstack((input_split_1, input_split_2)), train_calls[2][0][0])
        np.testing.assert_array_equal(np.vstack((output_split_2, output_split_3)), train_calls[0][0][1])
        np.testing.assert_array_equal(np.vstack((output_split_1, output_split_3)), train_calls[1][0][1])
        np.testing.assert_array_equal(np.vstack((output_split_1, output_split_2)), train_calls[2][0][1])

        get_loss_calls = self.trainer._nn.get_loss.call_args_list
        np.testing.assert_array_equal(input_split_1, get_loss_calls[0][0][0])
        np.testing.assert_array_equal(input_split_2, get_loss_calls[1][0][0])
        np.testing.assert_array_equal(input_split_3, get_loss_calls[2][0][0])
        np.testing.assert_array_equal(output_split_1, get_loss_calls[0][0][1])
        np.testing.assert_array_equal(output_split_2, get_loss_calls[1][0][1])
        np.testing.assert_array_equal(output_split_3, get_loss_calls[2][0][1])

    def test_k_fold_cross_validation_invalid_k(self):
        # given
        input_vectors = np.array([
            [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]], [[5, 5]]
        ])
        output_vectors = np.array([
            [[1]], [[2]], [[3]], [[4]], [[5]]
        ])
        method = 'method'
        num_iterations = 100
        alpha = 0.1
        k = 6
        self.trainer.train = MagicMock()
        self.trainer._nn.get_loss.return_value = 1

        # then
        with self.assertRaises(ValueError):
            # when
            self.trainer.k_fold_cross_validation(input_vectors, output_vectors, k, method, num_iterations, alpha)

    def test_k_fold_cross_validation_invalid_vectors(self):
        # given
        input_vectors = np.array([
            [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]]
        ])
        output_vectors = np.array([
            [[1]], [[2]], [[3]], [[4]], [[5]]
        ])
        method = 'method'
        num_iterations = 100
        alpha = 0.1
        k = 2
        self.trainer.train = MagicMock()
        self.trainer._nn.get_loss.return_value = 1

        # then
        with self.assertRaises(ValueError):
            # when
            self.trainer.k_fold_cross_validation(input_vectors, output_vectors, k, method, num_iterations, alpha)


if __name__ == '__main__':
    unittest.main()
