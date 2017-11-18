from copy import deepcopy

class Trainer(object):
    def __init__(self, nn):
        self._nn = nn

    def train(self, input_vectors, output_vectors, method='stochastic',
              num_iterations=1000, alpha=0.1):
        if method is 'stochastic':
            self._stochastic_gradient_descent(input_vectors, output_vectors,
                                              num_iterations, alpha)
        elif method is 'standard':
            self._standard_gradient_descent(input_vectors, output_vectors,
                                            num_iterations, alpha)
        else:
            raise ValueError('Invalid gradient descent method')

    def _stochastic_gradient_descent(self, input_vectors, output_vectors,
                                     num_iterations, alpha):
        for _ in range(num_iterations):
            for input_vector, output_vector in zip(input_vectors, output_vectors):
                weights = self._nn.get_weights()
                gradients = self._nn.get_gradient(input_vector, output_vector)
                for weight, gradient in zip(weights, gradients):
                    weight -= alpha * gradient
                self._nn.set_weights(weights)

    def _standard_gradient_descent(self, input_vectors, output_vectors,
                                   num_iterations, alpha):
        for _ in range(num_iterations):
            # We copy the weights in order to update them only after we sum the gradients.
            weights = deepcopy(self._nn.get_weights())
            for input_vector, output_vector in zip(input_vectors, output_vectors):
                gradients = self._nn.get_gradient(input_vector, output_vector)
                for weight, gradient in zip(weights, gradients):
                    weight -= alpha * gradient
            self._nn.set_weights(weights)
