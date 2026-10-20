from experiments.config import Config, Experiment, sweep
from nn.activation_functions import Linear, ReLU, Sigmoid, Softmax, Tanh
from nn.loss_function import BCE, CCE, MSE

# every experiment the project can run, by name: python main.py run <name>
# a new comparison is a new entry here: a base config plus the variants to try,
# sweep(param=[...]) builds the variants, or write them by hand as {"label": {param: value, ...}}

REGRESSION = Config(dataset="regression_cube_100", hidden=(16,), activation=Tanh, learning_rate=0.05, epochs=500)
CLASSIFICATION = Config(dataset="classification_three_gauss_100", hidden=(8,), activation=Sigmoid, learning_rate=0.1, epochs=300)

ARCHITECTURES1 = [(1,), (2,), (8,), (16,), (32,), (64,)]
ARCHITECTURES2 = [(2,2), (8,8), (2,4), (8,32),(32,8), (16, 16),  (32, 32), (64, 64)]
ARCHITECTURES3 = [(2,2,2), (1,1,1,1), (2,2,2,2), (8,8,8,8),(16,8,4,2)]


EXPERIMENTS = [
    # single runs
    Experiment("regression", REGRESSION.with_(learning_rate=0.05), network_html=True),
    Experiment("classification", CLASSIFICATION.with_(hidden=(3, 4, 3)), network_html=True),

    # regression comparisons
    Experiment("regression_activation", REGRESSION, sweep(activation=[Tanh, Sigmoid, ReLU]),
               description="hidden layer activation, everything else fixed"),
    Experiment("regression_architecture", REGRESSION, sweep(hidden=ARCHITECTURES3),
               description="number and size of hidden layers, Tanh hidden activation"),
    Experiment("regression_learning_rate", REGRESSION, sweep(learning_rate=[0.005, 0.01, 0.05, 0.1])),
    Experiment("regression_dataset_size", REGRESSION,
               sweep(dataset=["regression_cube_100", "regression_cube_500", "regression_cube_1000"])),
    
    Experiment("regression_loss", REGRESSION, {
        "Linear + MSE": {"loss": MSE},
        "Sigmoid + MSE": {"output_activation": Sigmoid, "loss": MSE},
        "Sigmoid + BCE": {"output_activation": Sigmoid, "loss": BCE},
    }, seeds=(1, 2, 3), description="output layer and loss function for regression, y scaled to [0, 1]"),


    # classification comparisons
    Experiment("classification_activation", CLASSIFICATION, sweep(activation=[Tanh, Sigmoid, ReLU]),
               seeds=(1, 2, 3)),
    Experiment("classification_architecture", CLASSIFICATION, sweep(hidden=ARCHITECTURES1), seeds=(1, 2, 3)),
    Experiment("classification_loss", CLASSIFICATION, {
        "Softmax + CCE": {"output_activation": Softmax, "loss": CCE},
        "Softmax + MSE": {"output_activation": Softmax, "loss": MSE},
        "Sigmoid + MSE": {"output_activation": Sigmoid, "loss": MSE},
        "Sigmoid + BCE": {"output_activation": Sigmoid, "loss": BCE},
    }, seeds=(1, 2, 3), description="output layer and loss function, one-hot targets"),
    
]

EXPERIMENTS = {e.name: e for e in EXPERIMENTS}
