from dataclasses import dataclass, field, replace
from itertools import product

from nn.activation_functions import Tanh


@dataclass(frozen=True)
class Config:
    # everything one training run needs
    # input and output layer sizes come from the dataset, so only the hidden layers are set here
    dataset: str
    hidden: tuple = (16,)
    activation: type = Tanh
    output_activation: type = None  # None -> task default (Linear for regression, Softmax for classification)
    loss: type = None               # None -> task default (MSE for regression, CCE for classification)
    learning_rate: float = 0.05
    epochs: int = 500
    batch_size: int = 1
    seed: int = 42

    def with_(self, **changes):
        return replace(self, **changes)


@dataclass
class Experiment:
    # a base config and named variants of it; every variant is trained once per seed
    # with one variant and one seed this is just a single training run
    name: str
    base: Config
    variants: dict = field(default_factory=lambda: {"base": {}})  # label -> fields of base to change
    seeds: tuple = (42,)
    network_html: bool = False  # interactive weights picture for every variant (big file for many epochs)
    description: str = ""


def value_label(value):
    # short readable name of a config value, used in variant labels
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (tuple, list)):
        return "-".join(map(str, value)) if value else "none"
    return str(value)


def sweep(**params):
    # every combination of the given values, e.g. sweep(activation=[Tanh, ReLU], learning_rate=[0.01, 0.1])
    # gives 4 variants; with one parameter the label is just the value, with more it is "name=value, ..."
    names = list(params)
    variants = {}
    for values in product(*params.values()):
        if len(names) == 1:
            label = value_label(values[0])
        else:
            label = ", ".join(f"{n}={value_label(v)}" for n, v in zip(names, values))
        variants[label] = dict(zip(names, values))
    return variants
