import os
import numpy as np
from activation_functions import Sigmoid, Linear
from data_loader import load_xy, min_max
from loss_function import MSE
from mlp import MLP
from plot_results import plot_regression

LAYERS = [1, 3, 3, 1]
LEARNING_RATE = 0.2
EPOCHS = 500
BATCH_SIZE = 1
DATASET = "activation.100"

# results are saved under a name built from the configuration, so runs can be compared
CONFIG = f"layers{'-'.join(map(str, LAYERS))}_lr{LEARNING_RATE}_ep{EPOCHS}_bs{BATCH_SIZE}_{DATASET}"
RESULTS_DIR = "wyniki"
PLOTS_DIR = "wykresy"


def train_regression(x, y):
    mlp = MLP(LAYERS, Sigmoid(), Linear(), MSE(), LEARNING_RATE)
    mlp.train(EPOCHS, BATCH_SIZE, x, y)
    mlp.save("models/model_1_3_3_1_activation_100.json")
    return mlp


def predict_to_csv(x, y, model, out_path):
    # writes network output and target for every sample, one pair per line
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("predicted,target\n")
        for i in range(len(y)):
            a, z = model.feedforward(x[i])
            f.write(f"{float(a[-1][0])},{float(y[i][0])}\n")
    return out_path


def compare_regression(path):
    # reads the file written by predict_to_csv and returns the mean squared error and mean absolute error
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    predicted = data[:, 0]
    target = data[:, 1]
    mse = np.mean((predicted - target) ** 2)
    mae = np.mean(np.abs(predicted - target))
    print(f"MSE: {mse}, MAE: {mae}")
    return mse, mae


def run():
    x, y = load_xy("dane/projekt1/projekt1/regression/data.activation.train.100.csv")
    x, x_lo, x_hi = min_max(x)
    y, y_lo, y_hi = min_max(y)
    mlp = train_regression(x, y)

    # test set is scaled with the training set's lo and hi, not its own min and max
    x_test, y_test = load_xy("dane/projekt1/projekt1/regression/data.activation.test.100.csv")
    x_test, _, _ = min_max(x_test, x_lo, x_hi)
    y_test, _, _ = min_max(y_test, y_lo, y_hi)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = predict_to_csv(x_test, y_test, mlp, os.path.join(RESULTS_DIR, f"{CONFIG}.csv"))
    compare_regression(csv_path)
    plot_regression(x_test, csv_path, os.path.join(PLOTS_DIR, f"{CONFIG}.png"), title=CONFIG)
