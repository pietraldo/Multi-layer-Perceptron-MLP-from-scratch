import os
import numpy as np
from activation_functions import Sigmoid, Softmax
from data_loader import load_xy, min_max
from datasets import dataset_paths
from loss_function import CCE
from mlp import MLP
from plot_results import plot_classification, plot_loss

LAYERS = [2, 3,4, 3, 3]
LEARNING_RATE = 0.1
EPOCHS = 300
BATCH_SIZE = 1
DATASET = "classification_three_gauss_100"  # name from datasets.py
N_CLASSES = 3

# results are saved under a name built from the configuration, so runs can be compared
CONFIG = f"layers{'-'.join(map(str, LAYERS))}_lr{LEARNING_RATE}_ep{EPOCHS}_bs{BATCH_SIZE}_{DATASET}"
RESULTS_DIR = "wyniki"
PLOTS_DIR = "wykresy"
MODELS_DIR = "models"


def to_one_hot(labels):
    index = labels[:, 0].astype(int) - 1
    onehot = np.zeros((len(index), N_CLASSES))
    onehot[np.arange(len(index)), index] = 1
    return onehot


def train_classification(x, y):
    mlp = MLP(LAYERS, Sigmoid(), Softmax(), CCE(), LEARNING_RATE)
    losses = mlp.train(EPOCHS, BATCH_SIZE, x, y)
    mlp.save(os.path.join(MODELS_DIR, f"{CONFIG}.json"))
    return mlp, losses


def predict_to_csv(x, labels, model, out_path):
    # writes predicted and true class for every sample, classes as 1 and 2 like in the data
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("predicted,target\n")
        for i in range(len(x)):
            a, _ = model.feedforward(x[i])
            f.write(f"{int(np.argmax(a[-1])) + 1},{int(labels[i, 0])}\n")
    return out_path


def compare_classification(path):
    # reads the file written by predict_to_csv and returns the share of correct predictions
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    accuracy = np.mean(data[:, 0] == data[:, 1])
    print(f"accuracy: {accuracy:.2%}")
    return accuracy


def run():
    train_path, test_path = dataset_paths(DATASET)

    x, labels = load_xy(train_path)
    x, x_lo, x_hi = min_max(x)
    y = to_one_hot(labels)
    mlp, losses = train_classification(x, y)

    # test set is scaled with the training set's lo and hi, like in regression
    x_test, labels_test = load_xy(test_path)
    x_test, _, _ = min_max(x_test, x_lo, x_hi)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = predict_to_csv(x_test, labels_test, mlp, os.path.join(RESULTS_DIR, f"{CONFIG}.csv"))
    compare_classification(csv_path)
    plot_classification(x_test, csv_path, os.path.join(PLOTS_DIR, f"{CONFIG}.png"), title=CONFIG)
    plot_loss(losses, os.path.join(PLOTS_DIR, f"{CONFIG}_loss.png"), title=CONFIG)
