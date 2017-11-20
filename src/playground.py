import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from trainer import Trainer

if __name__ == '__main__':
    nn = NeuralNetwork((1, 10, 10, 1))

    X = np.random.uniform(0, 2 * np.pi, 100).reshape(-1, 1, 1)
    y = 0.5 * (1 + 0.8 * np.sin(X))
    Trainer(nn).train(X / (2 * np.pi), y, method='BFGS')

    xx = 2 * np.pi * np.array(range(100)) / 100.0
    yy = ((2 * nn.predict(xx / (2 * np.pi)) - 1) / 0.8).reshape(-1)
    plt.plot(xx, np.sin(xx), xx, yy)
    plt.show()
