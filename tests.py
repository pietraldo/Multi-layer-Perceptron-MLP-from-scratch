from mlp import MLP
from activation_functions import *
import numpy as np
from loss_function import MSE

def test_feedforward():
    
    mlp = MLP([1,2,1], Sigmoid(), MSE(), )
    
    # mlp.weights[0] = np.array([[0.5],  [-0.3]])
    # mlp.weights[1] = np.array([[0.8, -0.4]])
    # mlp.biases[1] = np.array([0.1, 0.2])
    # mlp.biases[2] = np.array([0.1])

    input_data = np.array([1])
    expected_output = np.array([0.605])

    output = mlp.feedforward(input_data)
    print(output)
    assert np.isclose(output[0], expected_output[0], 0.01)

def test_forward_backward():
    mlp = MLP([1,2,1], Sigmoid(), MSE(), 1)
    
    mlp.weights[0] = np.array([[0.5],  [-0.3]])
    mlp.weights[1] = np.array([[0.8, -0.4]])
    mlp.biases[1] = np.array([0.1, 0.2])
    mlp.biases[2] = np.array([0.1])

    input_data = np.array([1])
    expected_result = np.array([1])
    expected_output = np.array([0.605])

    a,z = mlp.feedforward(input_data)
    mlp.print()
    
    assert np.isclose(a[2], expected_output[0], 0.01)
    
    mlp.backpropagation(a,z,expected_result)
    
    mlp.print()