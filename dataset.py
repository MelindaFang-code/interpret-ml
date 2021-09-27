import pandas as pd
from data_utils import compute_indices


class Dataset(object):
    """
    Represents datasets, containing train, val, and test for X, y
    """
    def __init__(self, X, y, train_prop, valid_prop):
        n = X.shape[0]
        train_indices, val_indices, test_indices = compute_indices(n, train_prop, valid_prop)

        print(f"Number train examples: {train_indices.shape[0]} \n"
              f"Number validation examples: {val_indices.shape[0]} \n"
              f"Number of test examples: {test_indices.shape[0]} ")

        self.X_train = X[train_indices]
        self.y_train = y[train_indices]

        self.X_valid = X[val_indices]
        self.y_valid = y[val_indices]

        self.X_test = X[test_indices]
        self.y_test = y[test_indices]


def load_dataset(dataset_name: str, train_prop: float, valid_prop: float) -> Dataset:
    """
    Main dataset loading function, which returns dataset in form of numpy arrays (X,y)
    """
    if dataset_name == 'taiwan_credit_risk':
        dataset = load_taiwan_credit_risk(train_prop, valid_prop)
    elif dataset_name == 'ames_housing':
        dataset = load_ames_housing(train_prop, valid_prop)
    elif dataset_name == 'civil_war_onset':
        dataset = load_civil_war_onset(train_prop, valid_prop)
    else:
        raise ValueError('Invalid dataset name')
    return dataset


def load_taiwan_credit_risk(train_prop, valid_prop) -> Dataset:
    df = pd.read_csv('datasets/taiwan_credit_risk.csv')
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    dataset = Dataset(X, y, train_prop, valid_prop)
    return dataset


def load_ames_housing(train_prop, valid_prop) -> Dataset:
    train = pd.read_csv('datasets/ames_housing/ames_housing_training.csv')
    test = pd.read_csv('datasets/ames_housing/ames_housing_test.csv')

    raise NotImplementedError


def load_civil_war_onset(train_prop, valid_prop) -> Dataset:
    # Dataset from Ethnicity, Insurgency, and Civil War by Fearon and Laitin 2003
    # Used by https://journals.sagepub.com/doi/pdf/10.1177/2053168020905487

    df = pd.read_csv('datasets/fearson_laitin.csv')
    X = df['onset']
    y = df.loc[:, df.columns != 'onset']

    raise NotImplementedError 

