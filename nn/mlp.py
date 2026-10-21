import json
import os
import numpy as np
from .activation_functions import *
from .loss_function import MSE, CCE, BCE
from .learning_step_update import Constant, StepDecay, ExponentialDecay, InverseTimeDecay, ReduceOnPlateau

# names stored in the model file -> classes used to rebuild the model
ACTIVATIONS = {"Sigmoid": Sigmoid, "Tanh": Tanh, "Linear": Linear, "Softmax": Softmax, "ReLU": ReLU}
LOSS_FUNCTIONS = {"MSE": MSE, "CCE": CCE, "BCE": BCE}
LEARNING_STEP_UPDATES = {"Constant": Constant, "StepDecay": StepDecay, "ExponentialDecay": ExponentialDecay,
                         "InverseTimeDecay": InverseTimeDecay, "ReduceOnPlateau": ReduceOnPlateau}

class MLP:

    weights = []
    biases = []
    num_layers = 0

    def __init__(self, layers, activation, output_activation, loss_function, learning_step, seed=42, learning_step_update=None):
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.num_layers = len(layers)
        self.loss_function = loss_function
        # learning step given by the user, learning_step is the current one and changes during training
        self.initial_learning_step = learning_step
        self.learning_step = learning_step
        # how the learning step changes between epochs, by default it stays the same
        self.learning_step_update = learning_step_update if learning_step_update is not None else Constant()
        # counts epochs across train calls, so step based updates continue correctly after loading a model
        self.epochs_trained = 0
        self.weights = []
        self.biases = []

        # one seeded generator for the initial weights and for shuffling, so the same seed gives the same run
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        rng = self.rng

        # initializing random weights
        for i in range(0, self.num_layers - 1):
            matrix = rng.normal(0, np.sqrt(1 / layers[i]), (layers[i + 1], layers[i]))
            self.weights.append(matrix)

        # initializing random biases, input layer has no bias so biases[i] belongs to layer i+1
        for size in layers[1:]:
            self.biases.append(rng.random(size))
            
    def train(self, epoch, batch_size, x, y, verbose=True):
        
        losses = []
        # learning step used in every epoch
        self.learning_steps = []
        # weights and biases before training and after every epoch, used to show how the network changed
        self.history = [self.snapshot()]
        for epoch_num in range(epoch):
            
            loss = 0
            
            # shuffle x, y
            perm = self.rng.permutation(len(x))
            x = x[perm]
            y = y[perm]
            
            grad_w_all = []
            grad_b_all = []
            for i in range(self.num_layers - 1):
                grad_w_all.append(np.zeros((self.layers[i+1], self.layers[i])))
                grad_b_all.append(np.zeros(self.layers[i+1]))
                
            for i in range(len(x)):
                a,z = self.feedforward(x[i])
                grad_w, grad_b = self.backpropagation(a,z, y[i])
                
                loss += self.loss_function.function(a[-1], y[i])
                
                for idx in range(len(grad_w)):
                    grad_w_all[idx] += grad_w[idx]
                    
                
                for idx in range(len(grad_b)):
                    grad_b_all[idx] += grad_b[idx]
                                                
                if (i+1) % batch_size == 0 or i + 1 == len(x):
                    for idx in range(len(grad_w)):
                        grad_w_all[idx] /= batch_size
                    for idx in range(len(grad_b)):
                        grad_b_all[idx] /= batch_size
                    
                    self.update_model_parameters(grad_w_all, grad_b_all)
                    
                    for arr in grad_w_all:
                        arr[:] = 0
                    for arr in grad_b_all:
                        arr[:] = 0
            
            if verbose:
                print(f"Epoch {epoch_num+1} loss: {loss} learning step: {self.learning_step}")
            losses.append(loss)
            self.learning_steps.append(self.learning_step)
            self.history.append(self.snapshot())

            self.epochs_trained += 1
            self.update_learning_step(loss)

        return losses

    def update_learning_step(self, loss):
        self.learning_step = self.learning_step_update.update(self.learning_step, self.epochs_trained, loss)

    def snapshot(self):
        # copy of the current weights and biases, as plain lists
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def feedforward(self, input_data):
        a = [np.array([]) for _ in range(self.num_layers)]
        z = [np.array([]) for _ in range(self.num_layers)]

        a[0] = input_data

        for idx in range(1, self.num_layers):
            z[idx] = self.weights[idx - 1] @ a[idx - 1] + self.biases[idx - 1]
            
            if idx + 1 == self.num_layers:
                a[idx] = self.output_activation.function(z[idx])
            else:
                a[idx] = self.activation.function(z[idx])

        return a, z

    def predict(self, x):
        # network output for every row of x, shape (n, output size)
        return np.array([self.feedforward(xi)[0][-1] for xi in x])

    def backpropagation(self, a, z, expected):
        grad_out = self.loss_function.derivative(a[self.num_layers - 1], expected)
        error_signal = self.output_activation.backward(z[self.num_layers - 1], a[self.num_layers - 1], grad_out)

        grad_w = [[] for _ in range(self.num_layers - 1)]
        grad_b = [[] for _ in range(self.num_layers - 1)]

        for idx in range(self.num_layers - 2, -1, -1):
            grad_w[idx] = np.outer(error_signal, a[idx])
            grad_b[idx] = error_signal

            if idx > 0:
                error_signal = (self.weights[idx].T @ error_signal) * self.activation.derivative(z[idx])
        return grad_w, grad_b
    
    def update_model_parameters(self, grad_w, grad_b):
        for idx in range(0, self.num_layers - 1):
            self.weights[idx] = self.weights[idx] - self.learning_step * grad_w[idx]
            self.biases[idx] = self.biases[idx] - self.learning_step * grad_b[idx]

    def save(self, path):
        # saves the whole model as readable JSON, weights and biases can be edited by hand
        data = {
            "layers": self.layers,
            "activation": type(self.activation).__name__,
            "output_activation": type(self.output_activation).__name__,
            "loss_function": type(self.loss_function).__name__,
            # initial step, so the file describes the model as it was configured
            "learning_step": self.initial_learning_step,
            # step for the next epoch, used to continue training after loading
            "current_learning_step": self.learning_step,
            "learning_step_update": {
                "name": type(self.learning_step_update).__name__,
                "params": self.learning_step_update.params(),
            },
            "epochs_trained": self.epochs_trained,
            "seed": self.seed,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # files saved before learning_step_update existed used a constant step
        update = data.get("learning_step_update", {"name": "Constant", "params": {}})
        model = MLP(
            data["layers"],
            ACTIVATIONS[data["activation"]](),
            # files saved before output_activation existed were all sigmoid
            ACTIVATIONS[data.get("output_activation", "Sigmoid")](),
            LOSS_FUNCTIONS[data["loss_function"]](),
            data["learning_step"],
            seed=data.get("seed", 42),
            learning_step_update=LEARNING_STEP_UPDATES[update["name"]](**update["params"]),
        )
        model.learning_step = data.get("current_learning_step", data["learning_step"])
        model.epochs_trained = data.get("epochs_trained", 0)
        weights = [np.array(w, dtype=float) for w in data["weights"]]
        biases = [np.array(b, dtype=float) for b in data["biases"]]

        # check shapes, so a hand-edited file with a wrong size fails here and not during training
        layers = data["layers"]
        for i, w in enumerate(weights):
            if w.shape != (layers[i + 1], layers[i]):
                raise ValueError(f"weights[{i}] has shape {w.shape}, expected {(layers[i + 1], layers[i])}")
        for i, b in enumerate(biases):
            if b.shape != (layers[i + 1],):
                raise ValueError(f"biases[{i}] has shape {b.shape}, expected {(layers[i + 1],)}")

        model.weights = weights
        model.biases = biases
        return model

    def print(self):
        print("\nWeights: ")
        for weight_layer in self.weights:
            print(weight_layer)

        print("Baiases: ")
        for bias_layer in self.biases:
            print(bias_layer)
        print("\n")
