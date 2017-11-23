import numpy as np

from neural_network import NeuralNetwork
from training import stochastic_gradient_descent, batch_gradient_descent
from validation import k_fold_cross_validation

import logging
from logging_setup import setup_logging


def test_stochastic_gradient_descent(nn, input_examples, output_examples):
    nn.reset_weights()
    stochastic_gradient_descent(nn, input_examples, output_examples)
    prediction = nn.predict(input_examples)
    logging.info('Prediction for stochastic:\n{}'.format(prediction))
    logging.info('L2 loss: {}'.format(nn.get_loss(input_examples, output_examples) / input_examples.shape[0]))


def test_batch_gradient_descent(nn, input_examples, output_examples):
    nn.reset_weights()
    batch_gradient_descent(nn, input_examples, output_examples)
    prediction = nn.predict(input_examples)
    logging.info('Prediction for batch:\n{}'.format(prediction))
    logging.info('L2 loss: {}'.format(nn.get_loss(input_examples, output_examples) / input_examples.shape[0]))


def test_k_fold(nn, input_examples, output_examples, k):
    def train(nn, input_examples, output_examples):
        stochastic_gradient_descent(nn, input_examples, output_examples)

    mean_loss = k_fold_cross_validation(nn, k, input_examples, output_examples, train)
    logging.info('Mean L2 loss (k-fold, k={}): {}'.format(k, mean_loss))


def main():
    setup_logging('info')
    nn = NeuralNetwork((2, 3, 1))
    input_examples = np.array([
        [[3, 5]], [[5, 1]], [[10, 2]]
    ])
    output_examples = np.array([
        [[0.75]], [[0.82]], [[0.93]]
    ])
    test_stochastic_gradient_descent(nn, input_examples, output_examples)
    test_batch_gradient_descent(nn, input_examples, output_examples)
    test_k_fold(nn, input_examples, output_examples, k=3)


if __name__ == '__main__':
    main()
