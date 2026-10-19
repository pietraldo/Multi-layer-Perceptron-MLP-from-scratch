import os
import numpy as np
from activation_functions import Sigmoid, Linear
from data_loader import load_xy, min_max
from datasets import dataset_paths
from loss_function import MSE
from mlp import MLP
from plot_network import plot_network_html
from plot_results import plot_loss, plot_regression
from run_output import new_run_dir

LAYERS = [1, 2,4,4, 3, 1]
LEARNING_RATE = 1
EPOCHS = 500
BATCH_SIZE = 10
SEED = 42  # the same seed gives the same weights, the same shuffling and the same results
DATASET = "regression_cube_100"

# results are saved under a name built from the configuration, so runs can be compared
CONFIG = f"layers{'-'.join(map(str, LAYERS))}_lr{LEARNING_RATE}_ep{EPOCHS}_bs{BATCH_SIZE}_seed{SEED}_{DATASET}"


def train_regression(x, y, out_dir):
    mlp = MLP(LAYERS, Sigmoid(), Linear(), MSE(), LEARNING_RATE, seed=SEED)
    losses = mlp.train(EPOCHS, BATCH_SIZE, x, y)
    mlp.save(os.path.join(out_dir, f"{CONFIG}.json"))
    return mlp, losses


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
    out_dir = new_run_dir(CONFIG)
    train_path, test_path = dataset_paths(DATASET)

    # x is used as it is, only y is scaled, so test x values outside the training range are not changed
    x, y = load_xy(train_path)
    y, y_lo, y_hi = min_max(y)
    mlp, losses = train_regression(x, y, out_dir)

    # test set y is scaled with the training set's lo and hi, not its own min and max
    x_test, y_test = load_xy(test_path)
    y_test, _, _ = min_max(y_test, y_lo, y_hi)

    csv_path = predict_to_csv(x_test, y_test, mlp, os.path.join(out_dir, f"{CONFIG}.csv"))
    compare_regression(csv_path)
    plot_regression(x_test, csv_path, os.path.join(out_dir, f"{CONFIG}.png"), title=CONFIG)
    plot_loss(losses, os.path.join(out_dir, f"{CONFIG}_loss.png"), title=CONFIG)
    plot_network_html(mlp.history, losses, LAYERS, os.path.join(out_dir, f"{CONFIG}_network.html"), title=CONFIG)
