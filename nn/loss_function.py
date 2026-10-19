from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def function(self, output, expected):
        pass
    @abstractmethod
    def derivative(self, output, expected):
        pass

class MSE(LossFunction):
    def function(self, output, expected):
        return np.sum((output - expected) ** 2) / 2
    def derivative(self, output, expected):
        return output - expected

class CCE(LossFunction):
    def function(self, output, expected):
        eps = 1e-15
        output = np.clip(output, eps, 1.0)
        return -np.sum(expected * np.log(output))

    def derivative(self, output, expected):
        # true derivative dL/da, the softmax jacobian is applied in the activation's backward
        eps = 1e-15
        output = np.clip(output, eps, 1.0)
        return -expected / output

class BCE(LossFunction):
    # binary cross-entropy, meant for a single sigmoid output
    def function(self, output, expected):
        eps = 1e-15
        output = np.clip(output, eps, 1 - eps)
        return -np.sum(expected * np.log(output) + (1 - expected) * np.log(1 - output))

    def derivative(self, output, expected):
        # true derivative dL/da, the sigmoid part is applied in the activation's backward
        eps = 1e-15
        output = np.clip(output, eps, 1 - eps)
        return (output - expected) / (output * (1 - output))