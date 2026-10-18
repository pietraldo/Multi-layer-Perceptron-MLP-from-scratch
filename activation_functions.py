from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction:
    @abstractmethod
    def function(self, x):
        pass
    def derivative(self, x):
        pass

# backward(z, a, grad_out) takes dL/da and returns dL/dz for this layer's activation
# for a layer whose outputs depend on each other (softmax) the derivative is a matrix,
# so the product with the upstream gradient is done here, not elementwise

class Sigmoid(ActivationFunction):
    def function(self, x):
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        return self.function(x)*(1-self.function(x))
    def backward(self, z, a, grad_out):
        return grad_out * a * (1 - a)

class Linear(ActivationFunction):
    def function(self, x):
        return x
    def derivative(self, x):
        return 1
    def backward(self, z, a, grad_out):
        return grad_out

class Softmax(ActivationFunction):
    def function(self, x):
        exp_x = np.exp(x - np.max(x))  # numerical stability
        return exp_x / np.sum(exp_x)
    def derivative(self, x):
        raise NotImplementedError("softmax derivative is a matrix, use backward")
    def backward(self, z, a, grad_out):
        # jacobian is diag(a) - a a^T, so J @ grad_out = a * (grad_out - sum(grad_out * a))
        return a * (grad_out - np.sum(grad_out * a))