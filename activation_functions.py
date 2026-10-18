from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction:
    @abstractmethod
    def function(self, x):
        pass
    def derivative(self, x):
        pass

class Sigmoid(ActivationFunction):
    def function(self, x):
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        return self.function(x)*(1-self.function(x))
    
class Linear(ActivationFunction):
    def function(self, x):
        return x
    def derivative(self, x):
        return 1