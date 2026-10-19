import numpy as np
from numpy.testing import assert_allclose

from nn.mlp import MLP

MODEL_PATH = "./tests/models/model_z_wykladu.json"
TOL = 1e-4  # values from the lecture are rounded to 4 decimals


def test_forward_backward():
    mlp = MLP.load(MODEL_PATH)
    input_data = np.array([1.0])
    target = np.array([1.0])

    # forward pass
    a, z = mlp.feedforward(input_data)
    assert_allclose(z[1], [0.6, -0.1], atol=TOL)
    assert_allclose(z[2], [0.4265], atol=TOL)
    assert_allclose(a[1], [0.6457, 0.4750], atol=TOL)
    assert_allclose(a[2], [0.6050], atol=TOL)
    assert_allclose(mlp.loss_function.function(a[-1], target), 0.078, atol=1e-3)

    # one backpropagation step with learning_step = 1
    grad_w, grad_b = mlp.backpropagation(a, z, target)
    
    mlp.update_model_parameters(grad_w, grad_b)
    assert_allclose(mlp.weights[0], [[0.5173], [-0.3094]], atol=TOL)
    assert_allclose(mlp.biases[0], [0.1173, 0.1906], atol=TOL)
    assert_allclose(mlp.weights[1], [[0.8609, -0.3552]], atol=TOL)
    assert_allclose(mlp.biases[1], [0.1944], atol=TOL)

    # loss must go down after the update
    a, _ = mlp.feedforward(input_data)
    assert_allclose(mlp.loss_function.function(a[-1], target), 0.0636, atol=1e-3)
    print("Test 'test_forward_backward': passed")