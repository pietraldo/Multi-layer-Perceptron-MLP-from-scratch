import pickle
import numpy as np

class MLP:
    
    W = []
    b = []
    n = 0
    
    def __init__(self, layers, activision, lossFunction, learning_step):
        self.layers = layers
        self.activision = activision
        self.n = len(layers)
        self.lossFunction = lossFunction
        self.learning_step = learning_step
        self.W = []
        self.b = []

        rng = np.random.default_rng(42)
        
        # initializing random weights
        for i in range(0, self.n-1):
            matrix = rng.random((layers[i+1], layers[i]))
            self.W.append(matrix)
        
        # initializing random biases
        for size in layers:
            self.b.append(rng.random(size))
            
    def feedforward(self, input):
        a = [np.array([]) for _ in range(self.n)]
        z = [np.array([]) for _ in range(self.n)]
        
        a[0] = input
        
        for l in range(1, self.n):
            z[l] = self.W[l-1] @ a[l-1] + self.b[l]
            a[l] = self.activision.function(z[l])
        
        return a, z
    
    def backpropagation(self, a, z, expected):
        L = self.lossFunction.function(a[self.n - 1], expected)
        print(f"loss: {L}")
        error_signal = self.lossFunction.deriviative(a[self.n - 1], expected) * self.activision.deriviative(z[self.n - 1])
        print(error_signal)
        
        Gw = [[] for _ in range(self.n)]
        Gb = [[] for _ in range(self.n)]
        
        for l in range(self.n - 2, -1, -1):
            Gw[l] = np.outer(error_signal, a[l])
            Gb[l] = error_signal
            
            if l>0:
                error_signal = (self.W[l].T @ error_signal)  * self.activision.deriviative(z[l]) 
                
        for l in range(0, self.n-1):
            self.W[l] = self.W[l] - self.learning_step * Gw[l]
            self.b[l+1] = self.b[l+1] - self.learning_step * Gb[l]
    
    def save(self, path):
        # saves the whole model (weights, biases, layers, activation, loss, learning step)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def print(self):
        print("\nWeights: ")
        for wlayer in self.W:
            print(wlayer)
            
        print("Baiases: ")
        for blayer in self.b:
            print(blayer)
        print("\n")