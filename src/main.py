import numpy as np

from neural_network import NeuralNetwork
from training import stochastic_gradient_descent, batch_gradient_descent
from validation import k_fold_cross_validation


def test_stochastic_gradient_descent(nn, input_examples, output_examples):
    nn.reset_weights()
    stochastic_gradient_descent(nn, input_examples, output_examples)
    prediction = nn.predict(input_examples)
    print('Prediction for stochastic:\n{}'.format(prediction))
    print('L2 loss: {}'.format(nn.get_loss(input_examples, output_examples)))


def test_batch_gradient_descent(nn, input_examples, output_examples):
    nn.reset_weights()
    batch_gradient_descent(nn, input_examples, output_examples)
    prediction = nn.predict(input_examples)
    print('Prediction for batch:\n{}'.format(prediction))
    print('L2 loss: {}'.format(nn.get_loss(input_examples, output_examples)))


def test_k_fold(nn, input_examples, output_examples, k):
    mean_loss = k_fold_cross_validation(nn, k, input_examples, output_examples, stochastic_gradient_descent)
    print('Mean L2 loss (k-fold, k={}): {}'.format(k, mean_loss))


def main():
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
