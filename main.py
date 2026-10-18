import numpy as np
from tests import *
from mlp import *
from data_loader import load_xy, scale_y
from plot_results import plot_regression
# klasyfikacja i regresja
# liczba warstw neuronow funkcja aktywacji


def train(x, y):
    mlp = MLP([1,3,3,1], Sigmoid(), MSE(), 0.02)
    mlp.train(500, 10, x, y)
    mlp.save("model_1_3_3_1_activation_100.json")
    return mlp


def test(x, y, model, out_path="wyniki_test.csv"):
    # writes network output and target for every sample, one pair per line
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("predicted,target\n")
        for i in range(len(y)):
            a, z = model.feedforward(x[i])
            f.write(f"{float(a[-1][0])},{float(y[i][0])}\n")
    return out_path


def compare_regression(path):
    # reads the file written by test and returns the mean squared error and mean absolute error
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    predicted = data[:, 0]
    target = data[:, 1]
    mse = np.mean((predicted - target) ** 2)
    mae = np.mean(np.abs(predicted - target))
    print(f"MSE: {mse}, MAE: {mae}")
    return mse, mae


#test_forward_backward()

if __name__ == "__main__":
    x, y = load_xy("dane/projekt1/projekt1/regression/data.activation.train.100.csv")
    y, y_scale = scale_y(y)
    #mlp = train(x, y)
    mlp = MLP.load("models/model_1_3_3_1_activation_100.json")

    # test set is scaled with the same y_scale as the training set
    x_test, y_test = load_xy("dane/projekt1/projekt1/regression/data.activation.test.100.csv")
    y_test = y_test / y_scale
    test(x_test, y_test, mlp)
    compare_regression("wyniki_test.csv")
    plot_regression(x_test, "wyniki_test.csv", "wykres_regresja.png")

