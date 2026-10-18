import pickle
import numpy as np

class MLP:

    weights = []
    biases = []
    num_layers = 0

    def __init__(self, layers, activation, loss_function, learning_step):
        self.layers = layers
        self.activation = activation
        self.num_layers = len(layers)
        self.loss_function = loss_function
        self.learning_step = learning_step
        self.weights = []
        self.biases = []

        rng = np.random.default_rng(42)

        # initializing random weights
        for i in range(0, self.num_layers - 1):
            matrix = rng.random((layers[i + 1], layers[i]))
            self.weights.append(matrix)

        # initializing random biases
        for size in layers:
            self.biases.append(rng.random(size))

    def feedforward(self, input_data):
        a = [np.array([]) for _ in range(self.num_layers)]
        z = [np.array([]) for _ in range(self.num_layers)]

        a[0] = input_data

        for idx in range(1, self.num_layers):
            z[idx] = self.weights[idx - 1] @ a[idx - 1] + self.biases[idx]
            a[idx] = self.activation.function(z[idx])

        return a, z

    def backpropagation(self, a, z, expected):
        loss = self.loss_function.function(a[self.num_layers - 1], expected)
        print(f"loss: {loss}")
        error_signal = self.loss_function.derivative(a[self.num_layers - 1], expected) * self.activation.derivative(z[self.num_layers - 1])
        print(error_signal)

        grad_w = [[] for _ in range(self.num_layers)]
        grad_b = [[] for _ in range(self.num_layers)]

        for idx in range(self.num_layers - 2, -1, -1):
            grad_w[idx] = np.outer(error_signal, a[idx])
            grad_b[idx] = error_signal

            if idx > 0:
                error_signal = (self.weights[idx].T @ error_signal) * self.activation.derivative(z[idx])

        for idx in range(0, self.num_layers - 1):
            self.weights[idx] = self.weights[idx] - self.learning_step * grad_w[idx]
            self.biases[idx + 1] = self.biases[idx + 1] - self.learning_step * grad_b[idx]

    def save(self, path):
        # saves the whole model (weights, biases, layers, activation, loss function, learning step)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def print(self):
        print("\nWeights: ")
        for weight_layer in self.weights:
            print(weight_layer)

        print("Baiases: ")
        for bias_layer in self.biases:
            print(bias_layer)
        print("\n")
