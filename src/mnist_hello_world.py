from __future__ import division

import numpy as np

import mnist
from neural_network import NeuralNetwork
from training import stochastic_gradient_descent, batch_gradient_descent

import logging
from logging_setup import setup_logging


def test_mnist_one_hot(num_train_examples=-1, num_test_examples=-1):
    logging.info('Loading MNIST data.')
    train_input = convert_mnist_images(mnist.train_images()[:num_train_examples])
    train_output = convert_mnist_labels_one_hot(mnist.train_labels()[:num_train_examples])

    test_input = convert_mnist_images(mnist.test_images()[:num_test_examples])
    test_output = convert_mnist_labels_one_hot(mnist.test_labels()[:num_test_examples])

    nn = NeuralNetwork((784, 24, 32, 10), weight_decay=0.0)

    num_examples = train_input.shape[0]
    batch_size = 100

    def callback(iteration):
        if iteration % (num_examples // batch_size) == 0:
            training_prediction_rate = get_prediction_rate(nn, train_input, train_output)
            test_prediction_rate = get_prediction_rate(nn, test_input, test_output)
            training_loss = nn.get_loss(train_input, train_output)
            test_loss = nn.get_loss(test_input, test_output)
            print('{},{:.6f},{:.6f},{:.6f},{:.6f}'.format(iteration // (num_examples // batch_size),
                                                          training_prediction_rate, test_prediction_rate,
                                                          training_loss, test_loss))

    logging.info('MNIST training started.')
    print('epoch,training_accuracy,test_accuracy,training_loss,test_loss')
    stochastic_gradient_descent(nn, train_input, train_output, num_iterations=10000000,
                                learning_rate=0.1, momentum=0.1, batch_size=batch_size, callback=callback)


def convert_mnist_images(images):
    lst = []
    for image in images:
        lst.append(np.reshape(image, (1, image.size)))
    return normalize(np.array(lst))


def convert_mnist_labels_one_hot(labels):
    lst = []
    for label in labels:
        label_one_hot = -1 * np.ones(10)
        label_one_hot[label] = 1
        lst.append(np.array([label_one_hot]))
    return np.array(lst)


def normalize(data):
    return 2 * (data / 255.0) - 1


def get_prediction_rate(nn, test_input, test_output):
    prediction = nn.predict(test_input)
    diff = np.argmax(prediction, 2) - np.argmax(test_output, 2)
    error = np.count_nonzero(diff) / diff.size
    return 1.0 - error


if __name__ == '__main__':
    setup_logging('info')
    test_mnist_one_hot()
