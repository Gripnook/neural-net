from __future__ import division

import os
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator

from mnist_fully_connected import test_mnist_one_hot

plt.rcParams["font.family"] = "Times New Roman"
NUM_DATA_POINTS = 35
LOGS_DIRECTORY = 'csv'


def plot_directory(directory, x_label, y_label):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for filename in os.listdir(directory):
        x_range = pd.read_csv(os.path.join(directory, filename))[x_label][1:NUM_DATA_POINTS]
        y_range = pd.read_csv(os.path.join(directory, filename))[y_label][1:NUM_DATA_POINTS]
        splt = filename.split('.')[0].split('_')
        plt.plot(x_range, y_range, label='learning_rate={}%, layer_decay={}%'.format(splt[2], splt[4]))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    f.savefig('plots/{}_vs_{}.pdf'.format(y_label, x_label), bbox_inches='tight')


def plot_csv_simple(filename):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    csv_file = pd.read_csv('csv/{}.csv'.format(filename))
    x_range = csv_file['epoch']
    y_range = csv_file['test_accuracy']
    plt.plot(x_range, y_range)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    f.savefig('plots/{}.pdf'.format(filename), bbox_inches='tight')


def plot_test_accuracy_simple(y_range, filename):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_range = [i for i in range(100)]
    plt.plot(x_range, y_range)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    f.savefig('plots/{}.pdf'.format(filename), bbox_inches='tight')


def plot_test_accuracy_multiple(y_ranges, labels, filename):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_range = [i for i in range(100)]
    for y_range, label in zip(y_ranges, labels):
        plt.plot(x_range, y_range, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    f.savefig('plots/{}.pdf'.format(filename), bbox_inches='tight')


def plot_csv_multiple_zoom(csv_filenames, labels, filename, start_epoch=1, end_epoch=100):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_range = [i for i in range(start_epoch, end_epoch)]
    for csv_filename, label in zip(csv_filenames, labels):
        csv_file = pd.read_csv('csv/{}.csv'.format(csv_filename))
        y_range = csv_file['test_accuracy'][start_epoch:end_epoch]
        plt.plot(x_range, y_range, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    f.savefig('plots/{}.pdf'.format(filename), bbox_inches='tight')


def plot_network_size():
    lst_hidden_layers = [(100,), (200,), (300,), (100, 100), (200, 100), (300, 100)]
    accuracy_ranges = []
    labels = []
    for hidden_layers in lst_hidden_layers:
        test_accuracies = test_mnist_one_hot(
            hidden_layers=hidden_layers,
            csv_filename='network_size_{}'.format('_'.join(str(layer) for layer in hidden_layers)))
        accuracy_ranges.append(test_accuracies)
        labels.append('Hidden layers: {}'.format(hidden_layers))
    plot_test_accuracy_multiple(accuracy_ranges, labels, 'network_size')


def plot_network_size_csv():
    lst_hidden_layers = [(100,), (200,), (300,), (100, 100), (200, 100), (300, 100)]
    csv_filenames = ['network_size_{}'.format('_'.join(str(layer) for layer in hidden_layers))
                     for hidden_layers in lst_hidden_layers]
    labels = ['Hidden layer sizes: ({})'.format(', '.join(str(layer) for layer in hidden_layers))
              for hidden_layers in lst_hidden_layers]
    plot_csv_multiple_zoom(csv_filenames, labels, 'network_size', start_epoch=0)


def plot_network_size_csv_zoom():
    lst_hidden_layers = [(100,), (200,), (300,), (100, 100), (200, 100), (300, 100)]
    csv_filenames = ['network_size_{}'.format('_'.join(str(layer) for layer in hidden_layers))
                     for hidden_layers in lst_hidden_layers]
    labels = ['Hidden layer sizes: ({})'.format(','.join(str(layer) for layer in hidden_layers))
              for hidden_layers in lst_hidden_layers]
    plot_csv_multiple_zoom(csv_filenames, labels, 'network_size_zoom', start_epoch=1)


def plot_logistic_vs_tanh():
    test_accuracies_logistic = test_mnist_one_hot(sigmoid='logistic', learning_rate=0.04, csv_filename='logistic')
    test_accuracies_tanh = test_mnist_one_hot(sigmoid='tanh', csv_filename='tanh')
    plot_test_accuracy_multiple((test_accuracies_logistic, test_accuracies_tanh), ('logistic', 'tanh'),
                                'logistic_vs_tanh')


def plot_learning_rate():
    learning_rates = [0.01, 0.02, 0.05, 0.1]
    accuracy_ranges = []
    labels = []
    for learning_rate in learning_rates:
        test_accuracies = test_mnist_one_hot(learning_rate=learning_rate,
                                             csv_filename='learning_rate_{}'.format(int(learning_rate * 100)))
        accuracy_ranges.append(test_accuracies)
        labels.append('Learning rate: {}'.format(learning_rate))
    plot_test_accuracy_multiple(accuracy_ranges, labels, 'learning_rate')


def plot_learning_rate_csv_zoom():
    learning_rates = [0.01, 0.02, 0.05, 0.1]
    csv_filenames = ['learning_rate_{}'.format(int(learning_rate * 100)) for learning_rate in learning_rates]
    labels = ['Learning rate: {}'.format(learning_rate) for learning_rate in learning_rates]
    plot_csv_multiple_zoom(csv_filenames, labels, 'learning_rate_zoom')


def plot_batch_size():
    batch_sizes = [1, 10, 100]
    accuracy_ranges = []
    labels = []
    for batch_size in batch_sizes:
        test_accuracies = test_mnist_one_hot(batch_size=batch_size, csv_filename='batch_size_{}'.format(batch_size))
        accuracy_ranges.append(test_accuracies)
        labels.append('Batch size: {}'.format(batch_size))
    plot_test_accuracy_multiple(accuracy_ranges, labels, 'batch_size')


def plot_batch_size_csv_zoom():
    batch_sizes = [1, 10, 100]
    csv_filenames = ['batch_size_{}'.format(batch_size) for batch_size in batch_sizes]
    labels = ['Batch size: {}'.format(batch_size) for batch_size in batch_sizes]
    plot_csv_multiple_zoom(csv_filenames, labels, 'batch_size_zoom')


def plot_momentum():
    momenta = [i / 10 for i in range(11)]
    accuracy_ranges = []
    labels = []
    for momentum in momenta:
        test_accuracies = test_mnist_one_hot(momentum=momentum, csv_filename='momentum_{}'.format(int(momentum * 100)))
        accuracy_ranges.append(test_accuracies)
        labels.append('Momentum: {}'.format(momentum))
    plot_test_accuracy_multiple(accuracy_ranges, labels, 'momentum')


def plot_momentum_csv_zoom():
    momenta = [0.0, 0.3, 0.6, 0.9]
    csv_filenames = ['momentum_{}'.format(int(momentum * 100)) for momentum in momenta]
    labels = ['Momentum: {}'.format(momentum) for momentum in momenta]
    plot_csv_multiple_zoom(csv_filenames, labels, 'momentum')


def plot_layer_decay():
    layer_decays = [0.7, 0.8, 0.9, 0.99, 1]
    accuracy_ranges = []
    labels = []
    for layer_decay in layer_decays:
        test_accuracies = test_mnist_one_hot(
            layer_decay=layer_decay,
            csv_filename='layer_decay_{}'.format(int(layer_decay * 100)))
        accuracy_ranges.append(test_accuracies)
        labels.append('Layer decay: {}'.format(layer_decay))
    plot_test_accuracy_multiple(accuracy_ranges, labels, 'layer_decay')


def plot_layer_decay_csv_zoom():
    layer_decays = [0.7, 0.8, 0.9, 0.99, 1]
    csv_filenames = ['layer_decay_{}'.format(int(layer_decay * 100)) for layer_decay in
                     layer_decays]
    labels = ['Layer decay: {}'.format(layer_decay) for layer_decay in layer_decays]
    plot_csv_multiple_zoom(csv_filenames, labels, 'layer_decay', start_epoch=0)


if __name__ == '__main__':
    plot_logistic_vs_tanh()
    plot_learning_rate()
    plot_learning_rate_csv_zoom()
    plot_batch_size()
    plot_batch_size_csv_zoom()
    plot_momentum()
    plot_momentum_csv_zoom()
    plot_layer_decay()
    plot_layer_decay_csv_zoom()
    plot_network_size()
    plot_network_size_csv_zoom()
