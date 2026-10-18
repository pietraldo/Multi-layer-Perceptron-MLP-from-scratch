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
