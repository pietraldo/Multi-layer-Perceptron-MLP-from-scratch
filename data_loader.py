import numpy as np


def load_xy(path):
    # reads a csv with a header row, last column is the target y, all other columns are inputs x
    # returns x with shape (n, features) and y with shape (n, 1), ready for MLP.train
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    x = data[:, :-1]
    y = data[:, -1:]
    return x, y


def scale_y(y):
    # divides y by max(|y|) so all targets fall in [-1, 1], which fits the sigmoid output
    # returns the scaled y and the scale, multiply predictions by scale to get the original units
    scale = np.max(np.abs(y))
    return y / scale, scale
