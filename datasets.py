import os

DATA_DIR = os.path.join("dane", "projekt1", "projekt1")

# name -> (train file, test file), paths are relative to DATA_DIR
# the name goes into the result, plot and model file names, and DATASET in regression.py / classification.py is set to it
DATASETS = {
    "classification_simple_100":("classification/data.simple.train.100.csv", "classification/data.simple.test.100.csv"),
    "classification_simple_500":("classification/data.simple.train.500.csv", "classification/data.simple.test.500.csv"),
    "classification_simple_1000":("classification/data.simple.train.1000.csv", "classification/data.simple.test.1000.csv"),
    "classification_simple_10000":("classification/data.simple.train.10000.csv", "classification/data.simple.test.10000.csv"),
    "classification_three_gauss_100":("classification/data.three_gauss.train.100.csv", "classification/data.three_gauss.test.100.csv"),
    "classification_three_gauss_500":("classification/data.three_gauss.train.500.csv", "classification/data.three_gauss.test.500.csv"),
    "classification_three_gauss_1000":("classification/data.three_gauss.train.1000.csv", "classification/data.three_gauss.test.1000.csv"),
    "classification_three_gauss_10000":("classification/data.three_gauss.train.10000.csv", "classification/data.three_gauss.test.10000.csv"),
    "regression_activation_100":("regression/data.activation.train.100.csv", "regression/data.activation.test.100.csv"),
    "regression_activation_500":("regression/data.activation.train.500.csv", "regression/data.activation.test.500.csv"),
    "regression_activation_1000":("regression/data.activation.train.1000.csv", "regression/data.activation.test.1000.csv"),
    "regression_activation_10000":("regression/data.activation.train.10000.csv", "regression/data.activation.test.10000.csv"),
    "regression_cube_100":("regression/data.cube.train.100.csv", "regression/data.cube.test.100.csv"),
    "regression_cube_500":("regression/data.cube.train.500.csv", "regression/data.cube.test.500.csv"),
    "regression_cube_1000":("regression/data.cube.train.1000.csv", "regression/data.cube.test.1000.csv"),
    "regression_cube_10000":("regression/data.cube.train.10000.csv", "regression/data.cube.test.10000.csv"),
}


def dataset_paths(name):
    # returns the full paths of the train and test csv files for a dataset name
    train, test = DATASETS[name]
    return os.path.join(DATA_DIR, train), os.path.join(DATA_DIR, test)
