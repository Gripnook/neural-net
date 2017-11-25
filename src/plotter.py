import os
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator

NUM_DATA_POINTS = 35
LOGS_DIRECTORY = 'logs'


def plot(directory, x_label, y_label):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for filename in os.listdir(directory):
        x_range = pd.read_csv(os.path.join(directory, filename))[x_label][1:NUM_DATA_POINTS]
        y_range = pd.read_csv(os.path.join(directory, filename))[y_label][1:NUM_DATA_POINTS]
        splt = filename.split('.')[0].split('_')
        plt.plot(x_range, y_range,
                 label='learning_rate={}%, learning_decay={}%'.format(splt[2], splt[4]))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    f.savefig('plots/{}_vs_{}.pdf'.format(y_label, x_label))


if __name__ == '__main__':
    plot(LOGS_DIRECTORY, 'epoch', 'test_accuracy')
    plot(LOGS_DIRECTORY, 'epoch', 'test_loss')
