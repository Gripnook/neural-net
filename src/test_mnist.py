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


def test_mnist_one_hot(num_train_examples=100, num_test_examples=100):
    logging.info('Loading MNIST data.')
    train_input = convert_mnist_images(mnist.train_images()[:num_train_examples])
    train_output = convert_mnist_labels_one_hot(mnist.train_labels()[:num_train_examples])

    test_input = convert_mnist_images(mnist.test_images()[:num_test_examples])
    test_output = convert_mnist_labels_one_hot(mnist.test_labels()[:num_test_examples])

    nn = NeuralNetwork((784, 100, 10))
    trainer = Trainer(nn)
    logging.info('MNIST training started.')
    trainer.train(train_input, train_output, method='stochastic')
    logging.info('MNIST testing started.')
    logging.info('MNIST L2 loss: {}'.format(nn.get_loss(test_input, test_output)))


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
        label_one_hot = np.zeros(10)
        label_one_hot[label] = 1
        lst.append(np.array([label_one_hot]))
    return np.array(lst)


def normalize(data):
    return data / np.max(data)


if __name__ == '__main__':
    setup_logging('info')
    test_mnist_one_hot()
