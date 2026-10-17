from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction:
    @abstractmethod
    def function(self, x):
        pass
    def deriviative(self, x):
        pass
    
class Sigmoid(ActivationFunction):
    def function(self, x):
        return 1 / (1 + np.exp(-x))
    def deriviative(self, x):
        return self.function(x)*(1-self.function(x))
        