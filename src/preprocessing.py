from __future__ import division

import numpy as np

import mnist


def convert_mnist_images_logistic(images):
    data = flatten_input_data(images)
    return normalize_logistic(data)


def normalize_logistic(data):
    return data / 255.0


def convert_mnist_images_train_tanh(images):
    data = flatten_input_data(images)
    return normalize_tanh(data)


def convert_mnist_images_test_tanh(images, mean_shift, std_scale):
    data = flatten_input_data(images)
    data -= mean_shift
    data /= std_scale
    return data


def normalize_tanh(data):
    mean_shift = np.mean(data)
    data -= mean_shift
    std_scale = np.std(data)
    data /= std_scale
    return data, mean_shift, std_scale


def flatten_input_data(images):
    return np.array(images.reshape((images.shape[0], 1, images.shape[1] * images.shape[2])), dtype='float64')


def convert_mnist_labels_one_hot(labels, positive, negative):
    lst = []
    for label in labels:
        label_one_hot = negative * np.ones(10)
        label_one_hot[label] = positive
        lst.append(np.array([label_one_hot]))
    return np.array(lst)


def get_prediction_rate(nn, test_input, test_output):
    prediction = nn.predict(test_input)
    diff = np.argmax(prediction, 2) - np.argmax(test_output, 2)
    error = np.count_nonzero(diff) / diff.size
    return 1.0 - error
