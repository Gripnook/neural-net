from __future__ import division

import numpy as np


def stochastic_gradient_descent(nn, input_vectors, output_vectors, num_iterations=1000,
                                learning_rate=0.1, learning_decay=0.9, momentum=0.1, batch_size=100,
                                callback=lambda iteration: None):
    """
    Trains the neural network by using stochastic gradient descent
    with the given training examples.
    """

    delta_weights = [np.zeros(weight.shape) for weight in nn.get_weights()]
    learning_rates = get_learning_rates(nn, learning_rate, learning_decay)
    for iteration in range(num_iterations):
        # Get a random batch of examples.
        random_indices = np.random.randint(input_vectors.shape[0], size=batch_size)
        random_input_vectors = input_vectors[random_indices]
        random_output_vectors = output_vectors[random_indices]

        # Update the weights using the selected examples.
        update_weights(nn, random_input_vectors, random_output_vectors, delta_weights, learning_rates, momentum)

        # Let the user know that an iteration has completed.
        callback(iteration)


def batch_gradient_descent(nn, input_vectors, output_vectors, num_iterations=1000,
                           learning_rate=0.1, learning_decay=0.9, momentum=0.1, callback=lambda iteration: None):
    """
    Trains the neural network by using standard batch gradient
    descent with the given training examples.
    """

    delta_weights = [np.zeros(weight.shape) for weight in nn.get_weights()]
    learning_rates = get_learning_rates(nn, learning_rate, learning_decay)
    for iteration in range(num_iterations):
        # Update the weights using the examples.
        update_weights(nn, input_vectors, output_vectors, delta_weights, learning_rates, momentum)

        # Let the user know that an iteration has completed.
        callback(iteration)


def update_weights(nn, input_vectors, output_vectors, delta_weights, learning_rates, momentum):
    weights = nn.get_weights()
    gradients = nn.get_loss_gradient(input_vectors, output_vectors)
    for weight, gradient, delta_weight, learning_rate in zip(weights, gradients, delta_weights, learning_rates):
        delta_weight[:] = -learning_rate * gradient + momentum * delta_weight
        weight += delta_weight
    nn.set_weights(weights)


def get_learning_rates(nn, global_learning_rate, learning_decay):
    learning_rates = []
    running_learning_rate = global_learning_rate
    for i in range(nn.num_layers - 1):
        learning_rates.append(running_learning_rate)
        running_learning_rate *= learning_decay
    learning_rates *= 2
    return learning_rates
