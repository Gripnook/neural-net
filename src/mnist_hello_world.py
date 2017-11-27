from __future__ import division

import numpy as np

import mnist
from neural_network import NeuralNetwork
from training import stochastic_gradient_descent
from preprocessing import *


def test_mnist_one_hot(num_train_examples=-1, num_test_examples=-1):
    # Set the neural network parameters.
    layer_sizes = (784, 24, 32, 10)
    sigmoid = 'tanh'  # 'logistic' or 'tanh'
    weight_decay = 0.0

    print('Network Parameters')
    print('layer_sizes: {}, sigmoid: {}, weight_decay: {}'.format(layer_sizes, sigmoid, weight_decay))

    # Set the training parameters.
    num_iterations = 1000000
    learning_rate = 0.01
    learning_decay = 1.0
    momentum = 0.0
    batch_size = 100

    print('Training Parameters')
    print('num_iterations: {}, learning_rate: {}, learning_decay: {}, momentum: {}, batch_size: {}'.format(
        num_iterations, learning_rate, learning_decay, momentum, batch_size))

    print('')

    # Collect and preprocess the data.
    if sigmoid == 'logistic':
        train_input = convert_mnist_images_logistic(mnist.train_images()[:num_train_examples])
        train_output = convert_mnist_labels_one_hot(
            mnist.train_labels()[:num_train_examples], positive=0.9, negative=0.1)
        test_input = convert_mnist_images_logistic(mnist.test_images()[:num_test_examples])
        test_output = convert_mnist_labels_one_hot(mnist.test_labels()[:num_test_examples], positive=0.9, negative=0.1)
    elif sigmoid == 'tanh':
        train_input, mean_shift, std_scale = convert_mnist_images_train_tanh(mnist.train_images()[:num_train_examples])
        train_output = convert_mnist_labels_one_hot(
            mnist.train_labels()[:num_train_examples], positive=1.0, negative=-1.0)
        test_input = convert_mnist_images_test_tanh(mnist.test_images()[:num_test_examples], mean_shift, std_scale)
        test_output = convert_mnist_labels_one_hot(mnist.test_labels()[:num_test_examples], positive=1.0, negative=-1.0)
    else:
        raise ValueError('Invalid sigmoid function.')

    # Create and train the neural network.
    nn = NeuralNetwork(layer_sizes, sigmoid=sigmoid, weight_decay=weight_decay)

    num_examples = train_input.shape[0]

    def callback(iteration):
        if iteration % (num_examples // batch_size) == 0:
            training_prediction_rate = get_prediction_rate(nn, train_input, train_output)
            test_prediction_rate = get_prediction_rate(nn, test_input, test_output)
            training_loss = nn.get_loss(train_input, train_output)
            test_loss = nn.get_loss(test_input, test_output)
            print('{},{:.6f},{:.6f},{:.6f},{:.6f}'.format(iteration // (num_examples // batch_size),
                                                          training_prediction_rate, test_prediction_rate,
                                                          training_loss, test_loss))

    print('epoch,training_accuracy,test_accuracy,training_loss,test_loss')
    stochastic_gradient_descent(nn, train_input, train_output, num_iterations=num_iterations,
                                learning_rate=learning_rate, learning_decay=learning_decay,
                                momentum=momentum, batch_size=batch_size,
                                callback=callback)


if __name__ == '__main__':
    test_mnist_one_hot()
