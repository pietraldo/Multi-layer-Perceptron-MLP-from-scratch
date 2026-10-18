import numpy as np


def load_xy(path):
    # reads a csv with a header row, last column is the target y, all other columns are inputs x
    # returns x with shape (n, features) and y with shape (n, 1), ready for MLP.train
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    x = data[:, :-1]
    y = data[:, -1:]
    return x, y


def min_max(a, lo=None, hi=None):
    # scales each column of a to [0, 1], sigmoid can only output values in that range
    # pass lo and hi from the training set to scale the test set the same way
    # returns the scaled array and the lo, hi that were used
    if lo is None:
        lo = a.min(axis=0)
        hi = a.max(axis=0)
    return (a - lo) / (hi - lo), lo, hi
