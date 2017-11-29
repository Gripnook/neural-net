from __future__ import division

import random

import mnist

from csv_saver import save_rows_to_csv


def random_predict_mnist(num_epochs=100, csv_filename='random'):
    test_output = mnist.test_labels()
    num_test_examples = test_output.size
    rows = []
    header = 'epoch,test_accuracy'
    print(header)
    for epoch in range(num_epochs):
        correct = 0
        for label in test_output:
            random_choice = random.randint(0, 9)
            if label == random_choice:
                correct += 1
        test_accuracy = correct / num_test_examples
        print('{},{:.6f}'.format(epoch, test_accuracy))
        rows.append((epoch, test_accuracy))
    save_rows_to_csv(csv_filename, rows, header.split(','))


if __name__ == '__main__':
    random_predict_mnist()
