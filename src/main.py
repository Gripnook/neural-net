from __future__ import division

from Tkinter import *
from PIL import Image, ImageTk
from ast import literal_eval

import numpy as np
from scipy import misc
from scipy.ndimage.filters import gaussian_filter

import mnist
from neural_network import NeuralNetwork
from training import stochastic_gradient_descent
from preprocessing import *


class MNISTNeuralNetworkGUI(object):

    def __init__(self, master):
        self.master = master
        master.title('MNIST Neural Network')

        self.training_images = mnist.train_images()
        self.training_labels = mnist.train_labels()
        self.test_images = mnist.test_images()
        self.test_labels = mnist.test_labels()

        self.init_config_frame()
        self.init_test_frame()
        self.draw_canvas()

        self.create_network()

    def init_config_frame(self):
        self.config_frame = Frame(self.master)
        self.config_frame.grid(row=0, column=0)

        self.network_label = Label(self.config_frame, text='Network Parameters')
        self.network_label.grid(row=0, columnspan=2)

        self.layer_sizes_label = Label(self.config_frame, text='Hidden Layers:')
        self.layer_sizes_label.grid(row=1, column=0, sticky=W)
        self.layer_sizes_var = StringVar()
        self.layer_sizes_var.set('(24, 32)')
        self.layer_sizes_entry = Entry(self.config_frame, textvar=self.layer_sizes_var)
        self.layer_sizes_entry.grid(row=1, column=1, sticky=W)

        self.sigmoid_label = Label(self.config_frame, text='Sigmoid:')
        self.sigmoid_label.grid(row=2, column=0, sticky=W)
        self.sigmoid_var = StringVar()
        self.sigmoid_var.set('tanh')
        self.sigmoid_menu = OptionMenu(self.config_frame, self.sigmoid_var, 'logistic', 'tanh')
        self.sigmoid_menu.grid(row=2, column=1, sticky=W)

        self.weight_decay_label = Label(self.config_frame, text='Regularization:')
        self.weight_decay_label.grid(row=3, column=0, sticky=W)
        self.weight_decay_var = StringVar()
        self.weight_decay_var.set('0.0')
        self.weight_decay_entry = Entry(self.config_frame, textvar=self.weight_decay_var)
        self.weight_decay_entry.grid(row=3, column=1, sticky=W)

        self.create_network_button = Button(self.config_frame, text='Create Network', command=self.create_network)
        self.create_network_button.grid(row=4, columnspan=2)

        placeholder = Label(self.config_frame)
        placeholder.grid(row=5, columnspan=2)

        self.training_label = Label(self.config_frame, text='Training Parameters')
        self.training_label.grid(row=6, columnspan=2)

        self.num_iterations_label = Label(self.config_frame, text='Iterations:')
        self.num_iterations_label.grid(row=7, column=0, sticky=W)
        self.num_iterations_var = StringVar()
        self.num_iterations_var.set('1000')
        self.num_iterations_entry = Entry(self.config_frame, textvar=self.num_iterations_var)
        self.num_iterations_entry.grid(row=7, column=1, sticky=W)

        self.learning_rate_label = Label(self.config_frame, text='Learning Rate:')
        self.learning_rate_label.grid(row=8, column=0, sticky=W)
        self.learning_rate_var = StringVar()
        self.learning_rate_var.set('0.01')
        self.learning_rate_entry = Entry(self.config_frame, textvar=self.learning_rate_var)
        self.learning_rate_entry.grid(row=8, column=1, sticky=W)

        self.learning_decay_label = Label(self.config_frame, text='Layer Decay:')
        self.learning_decay_label.grid(row=9, column=0, sticky=W)
        self.learning_decay_var = StringVar()
        self.learning_decay_var.set('1.0')
        self.learning_decay_entry = Entry(self.config_frame, textvar=self.learning_decay_var)
        self.learning_decay_entry.grid(row=9, column=1, sticky=W)

        self.momentum_label = Label(self.config_frame, text='Momentum:')
        self.momentum_label.grid(row=10, column=0, sticky=W)
        self.momentum_var = StringVar()
        self.momentum_var.set('0.0')
        self.momentum_entry = Entry(self.config_frame, textvar=self.momentum_var)
        self.momentum_entry.grid(row=10, column=1, sticky=W)

        self.batch_size_label = Label(self.config_frame, text='Batch Size:')
        self.batch_size_label.grid(row=11, column=0, sticky=W)
        self.batch_size_var = StringVar()
        self.batch_size_var.set('100')
        self.batch_size_entry = Entry(self.config_frame, textvar=self.batch_size_var)
        self.batch_size_entry.grid(row=11, column=1, sticky=W)

        self.train_button = Button(self.config_frame, text='Train', command=self.train)
        self.train_button.grid(row=12, columnspan=2)

        placeholder = Label(self.config_frame)
        placeholder.grid(row=13, columnspan=2)

        self.init_validation_frame(row=14)

        placeholder = Label(self.config_frame)
        placeholder.grid(row=15, columnspan=2)

    def init_validation_frame(self, row):
        self.validation_frame = Frame(self.config_frame)
        self.validation_frame.grid(row=row, columnspan=2)

        self.validation_training_accuracy_label = Label(self.validation_frame, text='Training Accuracy:')
        self.validation_training_accuracy_label.grid(row=0, column=0, sticky=W)
        self.validation_training_accuracy_var = Label(self.validation_frame)
        self.validation_training_accuracy_var.grid(row=0, column=1, sticky=W)

        self.validation_training_loss_label = Label(self.validation_frame, text='Training Sum Squared Error:')
        self.validation_training_loss_label.grid(row=1, column=0, sticky=W)
        self.validation_training_loss_var = Label(self.validation_frame)
        self.validation_training_loss_var.grid(row=1, column=1, sticky=W)

        self.validation_test_accuracy_label = Label(self.validation_frame, text='Test Accuracy:')
        self.validation_test_accuracy_label.grid(row=2, column=0, sticky=W)
        self.validation_test_accuracy_var = Label(self.validation_frame)
        self.validation_test_accuracy_var.grid(row=2, column=1, sticky=W)

        self.validation_test_loss_label = Label(self.validation_frame, text='Test Sum Squared Error:')
        self.validation_test_loss_label.grid(row=3, column=0, sticky=W)
        self.validation_test_loss_var = Label(self.validation_frame)
        self.validation_test_loss_var.grid(row=3, column=1, sticky=W)

    def init_test_frame(self):
        self.canvas_data = np.array(self.test_images[0], dtype='float64')

        self.test_frame = Frame(self.master)
        self.test_frame.grid(row=0, column=1)

        self.canvas = Canvas(self.test_frame, width=28 * 10, height=28 * 10)
        self.canvas.grid(rowspan=4, column=0)
        self.canvas.bind('<Button-1>', self.paint_white)
        self.canvas.bind('<B1-Motion>', self.paint_white)
        self.canvas.bind('<Button-3>', self.paint_black)
        self.canvas.bind('<B3-Motion>', self.paint_black)

        self.reset_button = Button(self.test_frame, text='Reset', command=self.reset)
        self.reset_button.grid(row=0, column=1)

        self.blur_button = Button(self.test_frame, text='Blur', command=self.blur)
        self.blur_button.grid(row=1, column=1)

        self.sharpen_button = Button(self.test_frame, text='Sharpen', command=self.sharpen)
        self.sharpen_button.grid(row=2, column=1)

        self.test_button = Button(self.test_frame, text='Test', command=self.test)
        self.test_button.grid(row=3, column=1)

        placeholder = Label(self.test_frame)
        placeholder.grid(row=4, columnspan=2)

        self.init_results_frame(row=5)

        placeholder = Label(self.test_frame)
        placeholder.grid(row=6, columnspan=2)

    def init_results_frame(self, row):
        self.results_frame = Frame(self.test_frame)
        self.results_frame.grid(row=row, columnspan=2)

        self.digit_labels = []
        self.digits = []
        for i in range(10):
            self.digit_labels.append(Label(self.results_frame, text=str(i)))
            self.digit_labels[i].grid(row=0, column=i)
            self.digits.append(Canvas(self.results_frame, width=28, height=28))
            self.digits[i].grid(row=1, column=i)

    def create_network(self):
        # Set the neural network parameters.
        self.layer_sizes = tuple([784] + list(literal_eval(self.layer_sizes_var.get())) + [10])
        self.sigmoid = self.sigmoid_var.get()
        if not (self.sigmoid == 'logistic' or self.sigmoid == 'tanh'):
            raise ValueError('Invalid sigmoid function.')
        self.weight_decay = float(self.weight_decay_var.get())
        self.nn = NeuralNetwork(self.layer_sizes, sigmoid=self.sigmoid, weight_decay=self.weight_decay)

        # Collect and preprocess the data.
        if self.sigmoid == 'logistic':
            self.train_input = convert_mnist_images_logistic(self.training_images)
            self.train_output = convert_mnist_labels_one_hot(self.training_labels, positive=0.9, negative=0.1)
        elif self.sigmoid == 'tanh':
            self.train_input, self.mean_shift, self.std_scale = convert_mnist_images_train_tanh(self.training_images)
            self.train_output = convert_mnist_labels_one_hot(self.training_labels, positive=1.0, negative=-1.0)
        else:
            raise ValueError('Invalid sigmoid function.')

        self.test()
        self.validate()

    def train(self):
        # Set the training parameters.
        self.num_iterations = int(self.num_iterations_var.get())
        self.learning_rate = float(self.learning_rate_var.get())
        self.learning_decay = float(self.learning_decay_var.get())
        self.momentum = float(self.momentum_var.get())
        self.batch_size = int(self.batch_size_var.get())

        stochastic_gradient_descent(self.nn, self.train_input, self.train_output, num_iterations=self.num_iterations,
                                    learning_rate=self.learning_rate, learning_decay=self.learning_decay,
                                    momentum=self.momentum, batch_size=self.batch_size)

        self.test()
        self.validate()

    def test(self):
        if self.sigmoid == 'logistic':
            self.test_results = self.nn.predict(convert_mnist_images_logistic(np.array([self.canvas_data])))
        elif self.sigmoid == 'tanh':
            self.test_results = self.nn.predict(convert_mnist_images_test_tanh(
                np.array([self.canvas_data]), self.mean_shift, self.std_scale))
        else:
            raise ValueError('Invalid sigmoid function.')

        for result, digit in zip(self.test_results.reshape(-1), self.digits):
            if self.sigmoid == 'logistic':
                color = '#%02x%02x%02x' % (int(255 * np.clip((result - 0.1) / 0.8, 0, 1)), 0, 0)
            elif self.sigmoid == 'tanh':
                color = '#%02x%02x%02x' % (int(255 * np.clip((result + 1.0) / 2.0, 0, 1)), 0, 0)
            else:
                raise ValueError('Invalid sigmoid function.')
            digit.config(bg=color)

    def validate(self):
        if self.sigmoid == 'logistic':
            test_input = convert_mnist_images_logistic(self.test_images)
            test_output = convert_mnist_labels_one_hot(self.test_labels, positive=0.9, negative=0.1)
        elif self.sigmoid == 'tanh':
            test_input = convert_mnist_images_test_tanh(self.test_images, self.mean_shift, self.std_scale)
            test_output = convert_mnist_labels_one_hot(self.test_labels, positive=1.0, negative=-1.0)
        else:
            raise ValueError('Invalid sigmoid function.')

        training_prediction_rate = 100 * get_prediction_rate(self.nn, self.train_input, self.train_output)
        test_prediction_rate = 100 * get_prediction_rate(self.nn, test_input, test_output)
        training_loss = self.nn.get_loss(self.train_input, self.train_output)
        test_loss = self.nn.get_loss(test_input, test_output)

        self.validation_training_accuracy_var.config(text=('%.2f %%' % (training_prediction_rate)))
        self.validation_test_accuracy_var.config(text=('%.2f %%' % (test_prediction_rate)))
        self.validation_training_loss_var.config(text=('%.4f' % (training_loss)))
        self.validation_test_loss_var.config(text=('%.4f' % (test_loss)))

    def reset(self):
        self.canvas_data = np.zeros((28, 28))
        self.draw_canvas()

    def blur(self):
        self.canvas_data = gaussian_filter(self.canvas_data, sigma=0.5)
        self.draw_canvas()

    def sharpen(self):
        self.canvas_data = misc.imfilter(self.canvas_data, ftype='sharpen')
        self.draw_canvas()

    def paint_white(self, event):
        x = (int(self.canvas.canvasx(event.x)) - 1) // 10
        y = (int(self.canvas.canvasy(event.y)) - 1) // 10
        self.canvas_data[y, x] = 255
        self.draw_canvas()

    def paint_black(self, event):
        x = (int(self.canvas.canvasx(event.x)) - 1) // 10
        y = (int(self.canvas.canvasy(event.y)) - 1) // 10
        self.canvas_data[y, x] = 0
        self.draw_canvas()

    def draw_canvas(self):
        canvas_data = np.repeat(self.canvas_data, 10, axis=0)
        canvas_data = np.repeat(canvas_data, 10, axis=1)
        self.image = Image.fromarray(canvas_data)
        self.photo = ImageTk.PhotoImage(image=self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)


def main():
    root = Tk()
    gui = MNISTNeuralNetworkGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
