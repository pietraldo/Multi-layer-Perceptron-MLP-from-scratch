import os
from dataclasses import dataclass

import numpy as np

from data.datasets import dataset_paths
from data.loader import load_xy, min_max, standardize
from nn.activation_functions import Linear, Softmax
from nn.loss_function import MSE, CCE
from plotting.results import plot_classification, plot_regression

# everything that differs between regression and classification lives here:
# how the data is scaled, the default output layer, the metrics and the result plots


@dataclass
class TaskData:
    x: np.ndarray            # training inputs, scaled
    y: np.ndarray            # training targets, as the network should output them
    x_test_raw: np.ndarray   # test inputs before scaling, for plots
    x_test: np.ndarray       # test inputs, scaled with the training statistics
    y_test: np.ndarray       # test targets in the same form as y (regression) or class labels (classification)
    subsets: dict            # test subset name -> boolean mask over the test set
    classes: np.ndarray = None


class RegressionTask:
    name = "regression"
    default_output_activation = Linear
    default_loss = MSE
    metrics = {"mse": "lower", "mae": "lower"}  # metric -> which direction is better
    main_metric = "mse"

    def load(self, dataset):
        train_path, test_path = dataset_paths(dataset)

        # x is standardized to mean 0 and std 1, y is scaled to [0, 1]
        x, y = load_xy(train_path)
        x_min, x_max = x.min(axis=0), x.max(axis=0)
        x, x_mean, x_std = standardize(x)
        y, y_lo, y_hi = min_max(y)

        # test set is scaled with the training set's statistics, not its own
        x_test_raw, y_test = load_xy(test_path)
        x_test, _, _ = standardize(x_test_raw, x_mean, x_std)
        y_test, _, _ = min_max(y_test, y_lo, y_hi)

        # two test sets: only points inside the training x range, and the whole test set (includes extrapolation)
        in_range = np.all((x_test_raw >= x_min) & (x_test_raw <= x_max), axis=1)
        subsets = {"train_range": in_range, "whole_test": np.ones(len(x_test_raw), dtype=bool)}
        return TaskData(x, y, x_test_raw, x_test, y_test, subsets)

    def evaluate(self, model, data):
        # one dict per test subset, predictions are kept for saving
        predicted_all = model.predict(data.x_test)[:, 0]
        target_all = data.y_test[:, 0]
        results = []
        for subset, mask in data.subsets.items():
            predicted, target = predicted_all[mask], target_all[mask]
            results.append({
                "subset": subset,
                "points": int(mask.sum()),
                "mse": float(np.mean((predicted - target) ** 2)),
                "mae": float(np.mean(np.abs(predicted - target))),
                "predicted": predicted,
                "target": target,
            })
        return results

    def save_plots(self, result, data, out_dir, title):
        csv_path = write_predictions(result, os.path.join(out_dir, f"{result['subset']}_predictions.csv"))
        mask = data.subsets[result["subset"]]
        plot_regression(data.x_test_raw[mask], csv_path, os.path.join(out_dir, f"{result['subset']}.png"), title=title)


class ClassificationTask:
    name = "classification"
    default_output_activation = Softmax
    default_loss = CCE
    metrics = {"accuracy": "higher"}
    main_metric = "accuracy"

    def load(self, dataset):
        train_path, test_path = dataset_paths(dataset)

        x, labels = load_xy(train_path)
        x, x_mean, x_std = standardize(x)
        classes = np.unique(labels[:, 0])
        y = self.one_hot(labels[:, 0], classes)

        x_test_raw, labels_test = load_xy(test_path)
        x_test, _, _ = standardize(x_test_raw, x_mean, x_std)

        subsets = {"whole_test": np.ones(len(x_test_raw), dtype=bool)}
        return TaskData(x, y, x_test_raw, x_test, labels_test[:, 0], subsets, classes)

    @staticmethod
    def one_hot(labels, classes):
        # one output neuron per class, in the order of classes
        index = np.searchsorted(classes, labels)
        onehot = np.zeros((len(labels), len(classes)))
        onehot[np.arange(len(labels)), index] = 1
        return onehot

    def evaluate(self, model, data):
        predicted_all = data.classes[np.argmax(model.predict(data.x_test), axis=1)]
        results = []
        for subset, mask in data.subsets.items():
            predicted, target = predicted_all[mask], data.y_test[mask]
            results.append({
                "subset": subset,
                "points": int(mask.sum()),
                "accuracy": float(np.mean(predicted == target)),
                "predicted": predicted,
                "target": target,
            })
        return results

    def save_plots(self, result, data, out_dir, title):
        csv_path = write_predictions(result, os.path.join(out_dir, f"{result['subset']}_predictions.csv"))
        mask = data.subsets[result["subset"]]
        plot_classification(data.x_test_raw[mask], csv_path, os.path.join(out_dir, f"{result['subset']}.png"), title=title)


def write_predictions(result, path):
    # network output and target for every test point, one pair per line
    with open(path, "w", encoding="utf-8") as f:
        f.write("predicted,target\n")
        for p, t in zip(result["predicted"], result["target"]):
            f.write(f"{float(p)},{float(t)}\n")
    return path


def task_for(dataset):
    # the task is taken from the dataset name, see data/datasets.py
    if dataset.startswith("regression"):
        return RegressionTask()
    if dataset.startswith("classification"):
        return ClassificationTask()
    raise ValueError(f"cannot tell the task of dataset {dataset!r}, name must start with regression or classification")
