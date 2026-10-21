# Multi-layer Perceptron (MLP) from scratch

A multi-layer perceptron (feed-forward neural network) written from scratch in Python with NumPy, without any ML framework.
The project covers forward and backward propagation, several activation functions and loss functions, learning-rate schedules,
a series of comparison experiments on synthetic datasets, a full MNIST digit classifier, and a small GUI where you draw digits
and the network recognizes them.

## Features

- **MLP from scratch**: any number of hidden layers, weight initialization, backpropagation, mini-batch training
- **Activations**: Sigmoid, Tanh, ReLU, Linear, Softmax
- **Losses**: MSE, BCE, CCE (categorical cross-entropy)
- **Learning-rate schedules**: Constant, StepDecay, ExponentialDecay, InverseTimeDecay, ReduceOnPlateau
- **Experiments**: named comparisons (activation, architecture, learning rate, loss function, dataset size) with automatic reports and plots
- **MNIST training**: trains a 784-256-128-10 network, saves a model and plots after every epoch
- **Digit recognizer**: draw a digit with the mouse and see the predicted probabilities
- **Data viewer**: Tkinter window for browsing the classification and regression datasets

## Requirements

- Python 3.10 or newer
- Packages:

```bash
pip install numpy matplotlib pandas pillow
```

`tkinter` comes with the standard Python installer on Windows and macOS. On Linux install it with your package manager
(for example `sudo apt install python3-tk`).

## Project layout

```
main.py                    # entry point for tests and experiments
configs.py                 # definitions of all experiments
training_mnist_model.py    # MNIST training
digit_recognizer.py       # drawing GUI for recognizing digits
visualize_data.py          # dataset viewer (GUI)
tests.py                   # forward/backward check against the lecture example
tests/models/              # model from the lecture, used by tests.py
nn/                        # MLP, activations, losses
nn/learning_step_update.py # learning-rate schedules (Constant, StepDecay, ExponentialDecay, ...)
experiments/               # experiment runner, tasks, reports
plotting/                  # plots and the interactive network view
data/, mnist/              # dataset loaders
dane/                      # datasets, included in the repository (see below)
models/demo/               # demo model for the digit recognizer, included
models/mnist/              # MNIST models from training (created by training)
wyniki/                    # experiment results (created by main.py run)
```

## Data

**MNIST** (included): the four IDX files are in `dane/mnist_dataset/`. The loader expects each file inside a folder with the same name, and this is where they are already:

```
dane/mnist_dataset/
├── train-images-idx3-ubyte/train-images-idx3-ubyte
├── train-labels-idx1-ubyte/train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte/t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte
```

**Project datasets** (classification and regression CSV files, included): they are in `dane/projekt1/projekt1/classification/` and
`dane/projekt1/projekt1/regression/`. The viewer and the experiments read them from there.

Everything needed to run every command is in the repository: the datasets above and the demo model in `models/demo/`.
Nothing has to be downloaded or trained first.

**Run everything from the project root** (the folder with `main.py`). Paths are relative to the current directory.

## Quick start

```bash
# 1. check that forward and backward propagation match the lecture example
python main.py test

# 2. list all experiments
python main.py list

# 3. run one experiment
python main.py run regression_activation

# 4. train the MNIST model (10 epochs with default settings)
python training_mnist_model.py

# 5. draw digits (newest trained model, or the demo model if nothing was trained yet)
python digit_recognizer.py
```

## Command cheat sheet

Copy a command, run it from the project root. Every option is explained in the sections below.

### Setup

```bash
pip install numpy matplotlib pandas pillow
```

### Tests and listing

```bash
python main.py test
python main.py list
```

### Experiments (`main.py run`)

```bash
python main.py run regression
python main.py run classification
python main.py run regression_activation
python main.py run regression_architecture
python main.py run regression_learning_rate
python main.py run regression_dataset_size
python main.py run regression_loss
python main.py run classification_activation
python main.py run classification_architecture
python main.py run classification_loss
```

Options for any of them (they can be combined and placed after the names):

```bash
python main.py run classification_loss --seeds 1 2 3          # seeds instead of the ones in configs.py
python main.py run classification_loss --epochs 20            # epochs instead of the ones in configs.py
python main.py run classification_loss --seeds 1 --epochs 1   # very quick check
```

Several experiments at once:

```bash
python main.py run regression classification
python main.py run regression_activation regression_learning_rate --seeds 1 2 --epochs 50
python main.py run regression regression_activation regression_architecture regression_learning_rate regression_dataset_size regression_loss classification classification_activation classification_architecture classification_loss --seeds 1 --epochs 1
```

### MNIST training (`training_mnist_model.py`)

```bash
python training_mnist_model.py                                              # 10 epochs, batch 32, lr 0.1, seed 42
python training_mnist_model.py --epochs 20
python training_mnist_model.py --epochs 1 --limit 1000                      # quick check
python training_mnist_model.py --epochs 15 --batch-size 64 --learning-rate 0.05 --seed 7
python training_mnist_model.py --epochs 30 --batch-size 16 --learning-rate 0.1 --seed 1 --limit 20000
```

### Digit recognizer (`digit_recognizer.py`)

```bash
python digit_recognizer.py                                                  # newest trained model, or the demo model
python digit_recognizer.py models/demo/model.json                           # demo model
python digit_recognizer.py models/mnist/run_20261021_114501/epoch_010/model.json   # a specific trained model (use your own run folder)
```

### Dataset viewer (`visualize_data.py`)

```bash
python visualize_data.py
```

## 1. Tests and experiments (`main.py`)

```bash
python main.py test                       # forward/backward check, prints "passed" on success
python main.py list                      # experiments with number of variants, seeds and description
python main.py run <name> [<name> ...]   # run one or more experiments
python main.py run <name> --seeds 1 2 3  # override the seeds from configs.py
python main.py run <name> --epochs 20    # override the number of epochs (quick check)
```

Options of `run`:

| Option | Meaning |
|---|---|
| `name` (one or more) | experiment names from `python main.py list`; they are checked against `configs.py` |
| `--seeds N [N ...]` | random seeds to use instead of the ones set in `configs.py` |
| `--epochs N` | number of epochs to use instead of the one set in `configs.py` |

Several experiments can be run in one command, and the options apply to all of them:

```bash
python main.py run regression classification
python main.py run regression_activation regression_learning_rate --seeds 1 2 --epochs 50
python main.py run classification_activation classification_architecture classification_loss --seeds 1 2 3 --epochs 100
```

### Available experiments

| Name | Variants | Seeds | What it compares |
|---|---|---|---|
| `regression` | 1 | 1 | single regression run, Tanh, 16 hidden neurons |
| `classification` | 1 | 1 | single classification run, hidden layers 3-4-3 |
| `regression_activation` | 3 | 1 | hidden activation: Tanh, Sigmoid, ReLU |
| `regression_architecture` | 5 | 1 | hidden layers: `(2,2,2)`, `(1,1,1,1)`, `(2,2,2,2)`, `(8,8,8,8)`, `(16,8,4,2)` |
| `regression_learning_rate` | 4 | 1 | learning rate: 0.005, 0.01, 0.05, 0.1 |
| `regression_dataset_size` | 3 | 1 | dataset size: 100, 500, 1000 points |
| `regression_loss` | 3 | 3 | output and loss: Linear + MSE, Sigmoid + MSE, Sigmoid + BCE |
| `classification_activation` | 3 | 3 | hidden activation: Tanh, Sigmoid, ReLU |
| `classification_architecture` | 6 | 3 | hidden layers: `(1,)` ... `(64,)` |
| `classification_loss` | 4 | 3 | output and loss: Softmax + CCE, Softmax + MSE, Sigmoid + MSE, Sigmoid + BCE |

The number of variants and seeds shown above is the default from `configs.py`. The `--seeds` and `--epochs` options change it.

### Results

Each run creates its own folder `wyniki/<date_time>_<name>/` containing:

- `results.md`: report with the summary tables
- `summary.csv`, `runs.csv`: results as tables (one row per variant / per run)
- `loss_comparison.png`, `mse_comparison.png`, `mae_comparison.png` (and similar, depending on the task): comparison plots
- `runs/<variant>/`: for the first seed of every variant: `model.json`, `loss.png`, and the plots of the predictions.
  Single runs (`regression`, `classification`) also get `network.html`, an interactive view of the weights

## 2. MNIST training (`training_mnist_model.py`)

```bash
python training_mnist_model.py                                  # 10 epochs, batch 32, learning rate 0.1, seed 42
python training_mnist_model.py --epochs 20                      # more epochs
python training_mnist_model.py --epochs 5 --limit 1000          # quick check on the first 1000 training images
python training_mnist_model.py --batch-size 64 --learning-rate 0.05 --seed 7
python training_mnist_model.py --epochs 30 --batch-size 16 --learning-rate 0.1 --seed 1 --limit 20000
```

| Option | Default | Meaning |
|---|---|---|
| `--epochs N` | `10` | number of training epochs |
| `--batch-size N` | `32` | number of images per mini-batch |
| `--learning-rate X` | `0.1` | initial learning rate, decayed after every epoch (see below) |
| `--seed N` | `42` | random seed for weight initialization and shuffling |
| `--limit N` | all | use only the first `N` training images (the test set is always complete) |

Architecture: 784 inputs (28×28 pixels scaled to [0, 1]) → 256 ReLU → 128 ReLU → 10 Softmax, trained with CCE loss.

Learning rate: the step starts at `--learning-rate` and is multiplied by 0.75 after every epoch, never going below 0.0001
(`ExponentialDecay(0.75, 0.0001)`). With the default 0.1, the step is 0.1, 0.075, 0.056, ... in the first three epochs.

Every run creates a new folder `models/mnist/run_<date_time>/`. After each epoch it writes `epoch_NNN/` with:

- `model.json`: the model (used by the digit recognizer). Besides the weights and biases it stores the architecture,
  the loss and activation names, the initial and current learning step, the learning-rate schedule with its parameters,
  and the number of epochs trained
- `loss.png`, `accuracy.png`: train and test curves up to this epoch
- `history.csv`: loss and accuracy for every epoch so far

The console shows the loss and accuracy on the training and test sets after every epoch.

## 3. Digit recognizer (`digit_recognizer.py`)

```bash
python digit_recognizer.py                                                     # newest model from models/mnist, or models/demo
python digit_recognizer.py models/demo/model.json                              # the demo model (98.4% on the MNIST test set)
python digit_recognizer.py models/mnist/run_20261021_103253/epoch_010/model.json   # a specific trained model
```

| Argument | Meaning |
|---|---|
| `model` (optional) | path to a `model.json`; by default the last epoch of the newest training run is used, and `models/demo/model.json` when there is no training run yet |

Draw a digit with the left mouse button. The prediction is updated while you draw, and the bars show the probability of
each digit 0–9. The preview shows the 28×28 image the network receives.

| Control | Action |
|---|---|
| left mouse button (drag) | draw |
| right mouse button (drag) | erase |
| `C` | clear the canvas |
| `R` | reset |
| `Esc` | close the window |

You can also choose another `model.json` from the window.

## 4. Dataset viewer (`visualize_data.py`)

```bash
python visualize_data.py
```

A window opens. Choose the task (**Klasyfikacja** = classification or **Regresja** = regression), the dataset, the size and
the split (**train** or **test**). The window shows the plot and statistics of the data. **Zapisz wykres** saves the plot as a PNG file.
The viewer reads from `dane/projekt1/projekt1/`, so run it from the project root.

## Typical workflow

```bash
python main.py test                                              # 1. make sure the network computes correctly
python main.py run classification_loss --seeds 1 2 3 --epochs 100   # 2. compare variants
python training_mnist_model.py --epochs 15                       # 3. train on MNIST
python digit_recognizer.py                                       # 4. test it by drawing digits
```

## Notes

- Results and models are never overwritten: every run gets a new folder with the date and time in its name.
- `.gitignore` keeps the datasets (`dane/`) and the demo model (`models/demo/`) in the repository. Generated files are not:
  training runs (`models/mnist/run_*/`), experiment results (`wyniki/`), `.venv/` and `__pycache__/`. The top-level copies of the
  MNIST files and the original archives in `dane/` are also left out, because the same data is already extracted in the subfolders.
- Plots are saved to files (matplotlib with the `Agg` backend), so the training scripts work without opening windows.
