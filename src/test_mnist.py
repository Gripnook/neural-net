from __future__ import division

import logging
import numpy as np
import mnist

from neural_network import NeuralNetwork
from logging_setup import setup_logging
from trainer import Trainer


def test_mnist(num_train_examples=100, num_test_examples=100):
    logging.info('Loading MNIST data.')
    train_input = convert_mnist_images(mnist.train_images()[:num_train_examples])
    train_output = convert_mnist_labels(mnist.train_labels()[:num_train_examples])

    test_input = convert_mnist_images(mnist.test_images()[:num_test_examples])
    test_output = convert_mnist_labels(mnist.test_labels()[:num_test_examples])

    nn = NeuralNetwork((784, 10, 1))
    trainer = Trainer(nn)
    logging.info('MNIST training started.')
    trainer.train(train_input, train_output, method='stochastic')
    logging.info('MNIST testing started.')
    logging.info('MNIST L2 loss: {}'.format(nn.get_loss(test_input, test_output)))


def test_mnist_one_hot(num_train_examples=1000, num_test_examples=100):
    logging.info('Loading MNIST data.')
    train_input = convert_mnist_images(mnist.train_images()[:num_train_examples])
    train_output = convert_mnist_labels_one_hot(mnist.train_labels()[:num_train_examples])

    test_input = convert_mnist_images(mnist.test_images()[:num_test_examples])
    test_output = convert_mnist_labels_one_hot(mnist.test_labels()[:num_test_examples])

    nn = NeuralNetwork((784, 76, 24, 10))
    trainer = Trainer(nn)
    logging.info('MNIST training started.')
    for _ in range(100):
        trainer.train(train_input, train_output, method='stochastic', num_iterations=10)
        logging.info('MNIST training prediction rate: {}'.format(get_prediction_rate(nn, train_input, train_output)))
        logging.info('MNIST test prediction rate: {}'.format(get_prediction_rate(nn, test_input, test_output)))
        logging.info('MNIST training L2 loss: {}'.format(nn.get_loss(train_input, train_output)))
        logging.info('MNIST test L2 loss: {}'.format(nn.get_loss(test_input, test_output)))


def convert_mnist_images(images):
    lst = []
    for image in images:
        lst.append(np.reshape(image, (1, image.size)))
    return normalize(np.array(lst))


def convert_mnist_labels(labels):
    lst = []
    for label in labels:
        lst.append(np.array([[label]]))
    return np.array(lst)


def convert_mnist_labels_one_hot(labels):
    lst = []
    for label in labels:
        label_one_hot = 0.1 * np.ones(10)
        label_one_hot[label] = 0.9
        lst.append(np.array([label_one_hot]))
    return np.array(lst)


def normalize(data):
    return data / 255.0


def get_prediction_rate(nn, test_input, test_output):
    prediction = nn.predict(test_input)
    diff = np.argmax(prediction, 2) - np.argmax(test_output, 2)
    error = np.count_nonzero(diff) / diff.size
    return 1.0 - error


if __name__ == '__main__':
    setup_logging('info')
    test_mnist_one_hot()
