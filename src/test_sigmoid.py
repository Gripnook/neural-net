from __future__ import division

import unittest

import numpy as np

from neural_network import tanh_sigmoid, tanh_sigmoid_prime


class TestSigmoid(unittest.TestCase):
    def test_tanh_sigmoid_prime(self):
        x = 1.0
        sig = tanh_sigmoid(x)

        expected_sig_prime = 1.7159 * 2 * (1 - np.tanh(2 * x / 3) ** 2) / 3
        actual_sig_prime = tanh_sigmoid_prime(sig)

        self.assertAlmostEqual(expected_sig_prime, actual_sig_prime)


if __name__ == '__main__':
    unittest.main()
