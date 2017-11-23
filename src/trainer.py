import random
import numpy as np


class Trainer(object):
    def __init__(self, nn):
        self._nn = nn

    def stochastic_gradient_descent(self, input_vectors, output_vectors, num_iterations=1000,
                                    learning_rate=0.1, momentum=0.1, batch_size=100,
                                    callback=lambda iteration: None):
        """
        Trains the neural network by using stochastic gradient descent
        with the given training examples.
        """

        # Scale the learning rate according to the batch size.
        learning_rate /= batch_size

        delta_weights = [np.zeros(weight.shape) for weight in self._nn.get_weights()]
        for iteration in range(num_iterations):
            # Get a random batch of examples.
            random_indices = np.random.randint(input_vectors.shape[0], size=batch_size)
            random_input_vectors = input_vectors[random_indices,:,:]
            random_output_vectors = output_vectors[random_indices,:,:]

            # Update the weights using the selected examples.
            weights = self._nn.get_weights()
            gradients = self._nn.get_gradient(random_input_vectors, random_output_vectors)
            for weight, gradient, delta_weight in zip(weights, gradients, delta_weights):
                delta_weight[:] = -learning_rate * gradient + momentum * delta_weight
                weight += delta_weight
            self._nn.set_weights(weights)

            # Let the user know that an iteration has completed.
            callback(iteration)
    
    def batch_gradient_descent(self, input_vectors, output_vectors, num_iterations=1000,
                               learning_rate=0.1, momentum=0.1, callback=lambda iteration: None):
        """
        Trains the neural network by using standard batch gradient
        descent with the given training examples.
        """

        # Scale the learning rate according to the input size.
        learning_rate /= input_vectors.shape[0]

        delta_weights = [np.zeros(weight.shape) for weight in self._nn.get_weights()]
        for iteration in range(num_iterations):
            # Update the weights using the examples.
            weights = self._nn.get_weights()
            gradients = self._nn.get_gradient(input_vectors, output_vectors)
            for weight, gradient, delta_weight in zip(weights, gradients, delta_weights):
                delta_weight[:] = -learning_rate * gradient + momentum * delta_weight
                weight += delta_weight
            self._nn.set_weights(weights)

            # Let the user know that an iteration has completed.
            callback(iteration)

# def k_fold_cross_validation(nn, input_vectors, output_vectors, k, train):
#     if k > len(input_vectors) or k > len(output_vectors):
#         raise ValueError('k cannot be greater than the total number of input/output vector pairs.')
    
#     if len(input_vectors) != len(output_vectors):
#         raise ValueError('Must have same number of input and output vectors.')
    
#     losses = []
#     index_offset = int(round(len(input_vectors) / k))
#     for i in range(k):
#         # Split training/testing data
#         test_start_index = i * index_offset
#         test_end_index = (i + 1) * index_offset if i < k - 1 else len(input_vectors)
#         train_input = np.vstack((input_vectors[:test_start_index], input_vectors[test_end_index:]))
#         train_output = np.vstack((output_vectors[:test_start_index], output_vectors[test_end_index:]))
#         test_input = input_vectors[test_start_index:test_end_index]
#         test_output = output_vectors[test_start_index:test_end_index]
        
#         # Train
#         nn.reset_weights()
#         train(nn)

#         # Test
#         loss = nn.get_loss(test_input, test_output)
#         losses.append(loss)
#     return np.mean(losses)
