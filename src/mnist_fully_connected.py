from __future__ import division

import mnist

from csv_saver import save_rows_to_csv
from neural_network import NeuralNetwork
from preprocessing import *
from training import stochastic_gradient_descent


def test_mnist_one_hot(num_train_examples=-1, num_test_examples=-1, hidden_layers=(100,), sigmoid='tanh',
                       learning_rate=0.01, layer_decay=1.0, momentum=0.0, batch_size=100, num_epochs=100,
                       csv_filename=None, return_test_accuracies=True):
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
    layer_sizes = (784,) + hidden_layers + (10,)
    weight_decay = 0.0
    nn = NeuralNetwork(layer_sizes, sigmoid=sigmoid, weight_decay=weight_decay)

    num_examples = train_input.shape[0]
    num_iterations = (num_examples // batch_size) * num_epochs

    rows = None
    if csv_filename is not None:
        rows = []

    test_accuracies = None
    if return_test_accuracies:
        test_accuracies = []

    def callback(iteration):
        if iteration % (num_examples // batch_size) == 0:
            epoch = iteration // (num_examples // batch_size)
            training_prediction_accuracy = get_prediction_accuracy(nn, train_input, train_output)
            test_prediction_accuracy = get_prediction_accuracy(nn, test_input, test_output)
            training_loss = nn.get_loss(train_input, train_output)
            test_loss = nn.get_loss(test_input, test_output)
            print('{},{:.6f},{:.6f},{:.6f},{:.6f}'.format(epoch, training_prediction_accuracy, test_prediction_accuracy,
                                                          training_loss, test_loss))
            if csv_filename is not None:
                rows.append((epoch, training_prediction_accuracy, test_prediction_accuracy, training_loss, test_loss))
            if return_test_accuracies:
                test_accuracies.append(test_prediction_accuracy)

    print('Network Parameters')
    print('layer_sizes: {}, sigmoid: {}, weight_decay: {}'.format(layer_sizes, sigmoid, weight_decay))
    print('Training Parameters')
    print('num_iterations: {}, learning_rate: {}, layer_decay: {}, momentum: {}, batch_size: {}'.format(
        num_iterations, learning_rate, layer_decay, momentum, batch_size))
    print('')

    header = 'epoch,training_accuracy,test_accuracy,training_loss,test_loss'
    print(header)
    stochastic_gradient_descent(nn, train_input, train_output, num_iterations=num_iterations,
                                learning_rate=learning_rate, layer_decay=layer_decay,
                                momentum=momentum, batch_size=batch_size,
                                callback=callback)

    if csv_filename is not None:
        save_rows_to_csv(csv_filename, rows, header.split(','))

    if return_test_accuracies:
        return test_accuracies


if __name__ == '__main__':
    test_mnist_one_hot()
