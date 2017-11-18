import unittest

from neural_network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork((2, 5, 1))

    def test_predict(self):
        pass

    def test_get_weights(self):
        pass

    def test_set_weights(self):
        pass

    def test_get_gradient(self):
        pass


if __name__ == '__main__':
    unittest.main()
