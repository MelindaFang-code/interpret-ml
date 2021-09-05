import pandas as pd
from data_utils import compute_indices


class Dataset(object):
    def __init__(self, X, y, train_prop: float, valid_prop: float):
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
    else:
        raise ValueError('Invalid dataset name')
    return dataset


def load_taiwan_credit_risk(train_prop, valid_prop) -> Dataset:
    df = pd.read_csv('datasets/taiwan_credit_risk.csv')
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    dataset = Dataset(X, y, train_prop, valid_prop)
    return dataset
