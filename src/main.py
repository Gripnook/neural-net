import logging

import numpy as np

from neural_network import NeuralNetwork
from trainer import Trainer

LOGGING_LEVELS = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def setup_logging(level):
    """
    Set up logging, with the specified logging level.

    :param level: the logging level
    """
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=LOGGING_LEVELS[level])


def test_standard_gradient_descent(trainer, nn, input_examples, output_examples):
    trainer.train(input_examples, output_examples, method='standard')
    prediction = nn.predict(input_examples)
    logging.info('Prediction for standard: {}'.format(prediction))
    logging.info('L2 loss: {}'.format(nn.get_loss(input_examples, output_examples)))


def test_stochastic_gradient_descent(trainer, nn, input_examples, output_examples):
    trainer.train(input_examples, output_examples, method='stochastic')
    prediction = nn.predict(input_examples)
    logging.info('Prediction for stochastic: {}'.format(prediction))
    logging.info('L2 loss: {}'.format(nn.get_loss(input_examples, output_examples)))


def test_bfgs(trainer, nn, input_examples, output_examples):
    trainer.train(input_examples, output_examples, method='BFGS')
    prediction = nn.predict(input_examples)
    logging.info('Prediction for BFGS: {}'.format(prediction))
    logging.info('L2 loss: {}'.format(nn.get_loss(input_examples, output_examples)))


def main():
    setup_logging('info')
    nn = NeuralNetwork((2, 3, 1))
    trainer = Trainer(nn)
    input_examples = np.array([
        [[3, 5]], [[5, 1]], [[10, 2]]
    ])
    output_examples = np.array([
        [[0.75]], [[0.82]], [[0.93]]
    ])
    test_standard_gradient_descent(trainer, nn, input_examples, output_examples)
    test_stochastic_gradient_descent(trainer, nn, input_examples, output_examples)
    test_bfgs(trainer, nn, input_examples, output_examples)


if __name__ == '__main__':
    main()
