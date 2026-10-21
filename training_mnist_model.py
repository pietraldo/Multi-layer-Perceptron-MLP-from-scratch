import argparse
import csv
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # plots are only saved to files, no window is needed
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mnist.MnistDataloader import MnistDataloader
from nn.activation_functions import ReLU, Softmax, Sigmoid
from nn.loss_function import CCE
from nn.mlp import MLP
from plotting.comparison import style_axes
from plotting.results import SURFACE, INK, INK_SECONDARY, TARGET, PREDICTED
from nn.learning_step_update import ExponentialDecay

# architecture: 784 inputs (flattened 28x28 image) -> Hidden layers ReLU -> 10 softmax
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
LAYERS = [INPUT_SIZE, 256, 128, NUM_CLASSES]

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "dane", "mnist_dataset")
MODELS_DIR = os.path.join(ROOT, "models", "mnist")


# ---------------------------------------------------------------- data

def load_mnist(data_dir):
    # raw images 28x28 with values 0-255 and integer labels, pool_size=1 keeps the full resolution
    loader = MnistDataloader(
        os.path.join(data_dir, "train-images-idx3-ubyte", "train-images-idx3-ubyte"),
        os.path.join(data_dir, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte"),
        os.path.join(data_dir, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"),
        pool_size=1,
    )
    return loader.load_data()


def flatten_images(images):
    # list of 28x28 images -> matrix (n, 784), pixels scaled to [0, 1] so ReLU does not blow up
    return np.array(images, dtype=float).reshape(len(images), -1) / 255.0


def one_hot(labels, num_classes=NUM_CLASSES):
    # digit labels -> one-hot rows, the target format for softmax + cross-entropy
    labels = np.asarray(labels, dtype=int)
    encoded = np.zeros((len(labels), num_classes))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded


def prepare_data(data_dir):
    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir)
    return (flatten_images(x_train), one_hot(y_train)), (flatten_images(x_test), one_hot(y_test))


# ---------------------------------------------------------------- model

def build_model(learning_rate, seed):
    return MLP(
        LAYERS,
        activation=ReLU(),
        output_activation=Softmax(),
        loss_function=CCE(),
        learning_step=learning_rate,
        seed=seed,
        learning_step_update = ExponentialDecay(0.75, 0.0001)
    )


def evaluate(model, x, y):
    # mean cross-entropy and accuracy in percent on the given set
    outputs = model.predict(x)
    loss = np.mean([model.loss_function.function(o, t) for o, t in zip(outputs, y)])
    accuracy = 100 * np.mean(outputs.argmax(axis=1) == y.argmax(axis=1))
    return float(loss), float(accuracy)


def train_one_epoch(model, x, y, batch_size):
    # MLP.train shuffles the data itself and returns the summed loss, here turned into a mean per sample
    summed_loss = model.train(1, batch_size, x, y, verbose=False)[0]
    return summed_loss / len(x)


# ---------------------------------------------------------------- saving

def create_run_dir(models_dir):
    # every run gets its own folder, so older runs are never overwritten
    run_dir = os.path.join(models_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def epoch_dir(run_dir, epoch):
    path = os.path.join(run_dir, f"epoch_{epoch:03d}")
    os.makedirs(path, exist_ok=True)
    return path


def save_history(history, path):
    # one row per epoch, easy to open in a spreadsheet or compare between runs
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------------- plots

def plot_train_test(history, train_key, test_key, ylabel, title, out_path):
    # one line for the training set and one for the test set, x axis is the epoch number
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.8), facecolor=SURFACE)
    style_axes(ax)
    ax.plot(epochs, [h[train_key] for h in history], color=TARGET, linewidth=2, marker="o", markersize=4,
            label="zbiór treningowy")
    ax.plot(epochs, [h[test_key] for h in history], color=PREDICTED, linewidth=2, marker="o", markersize=4,
            label="zbiór testowy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
    ax.set_xlabel("epoka", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_loss(history, out_path):
    title = f"Funkcja straty (CCE), test = {history[-1]['test_loss']:.4f}"
    return plot_train_test(history, "train_loss", "test_loss", "średnia strata na próbkę", title, out_path)


def plot_accuracy(history, out_path):
    title = f"Dokładność, test = {history[-1]['test_accuracy']:.2f}%"
    return plot_train_test(history, "train_accuracy", "test_accuracy", "poprawnie sklasyfikowane [%]", title, out_path)


def save_epoch_results(model, history, run_dir):
    # model, plots and history so far, all in the folder of the current epoch
    folder = epoch_dir(run_dir, history[-1]["epoch"])
    model.save(os.path.join(folder, "model.json"))
    plot_loss(history, os.path.join(folder, "loss.png"))
    plot_accuracy(history, os.path.join(folder, "accuracy.png"))
    save_history(history, os.path.join(folder, "history.csv"))
    return folder


# ---------------------------------------------------------------- training loop

def train(model, train_data, test_data, epochs, batch_size, run_dir):
    x_train, y_train = train_data
    x_test, y_test = test_data
    history = []
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, x_train, y_train, batch_size)

        # both sets are evaluated after the epoch with the same weights, so the curves are comparable
        train_loss, train_accuracy = evaluate(model, x_train, y_train)
        test_loss, test_accuracy = evaluate(model, x_test, y_test)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        })

        folder = save_epoch_results(model, history, run_dir)
        print(f"epoch {epoch}/{epochs}  "
              f"loss train {train_loss:.4f} test {test_loss:.4f}  "
              f"accuracy train {train_accuracy:.2f}% test {test_accuracy:.2f}%  -> {folder}")
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="trening sieci 784-128-10 na MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="bierze tylko pierwsze N obrazów treningowych, do szybkiego sprawdzenia")
    return parser.parse_args()


def main():
    args = parse_args()

    print("loading MNIST...")
    train_data, test_data = prepare_data(DATA_DIR)
    if args.limit:
        train_data = (train_data[0][:args.limit], train_data[1][:args.limit])
    print(f"train {train_data[0].shape}, test {test_data[0].shape}")

    model = build_model(args.learning_rate, args.seed)
    run_dir = create_run_dir(MODELS_DIR)
    print(f"saving to {run_dir}")

    train(model, train_data, test_data, args.epochs, args.batch_size, run_dir)


if __name__ == "__main__":
    main()
