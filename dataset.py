import pandas as pd
from data_utils import compute_indices
import numpy as np
import math



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
        assert train_prop + valid_prop == 1
        dataset = load_ames_housing(valid_prop)
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


def process_ames(X):
    ''' Helper for preprocess ames housing. Should be improved'''
    df = X[['Neighborhood', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','OpenPorchSF','PoolArea','MoSold', 'YrSold', 'SaleType']]
    df = pd.get_dummies(df)
    # cyclical encoding for month
    df['MoSold_cos'] = np.cos(2 * math.pi * df["MoSold"] / df["MoSold"].max()) 
    df['MoSold_sin'] = np.sin(2 * math.pi * df["MoSold"] / df["MoSold"].max())
    df = df.drop('MoSold', axis=1)
    return df


def load_ames_housing(valid_prop) -> Dataset:
    train_full = pd.read_csv('datasets/ames_housing/ames_housing_training.csv', index_col=0)
    train = process_ames(train_full)
    test_full = pd.read_csv('datasets/ames_housing/ames_housing_test.csv', index_col=0)
    test = process_ames(test_full)
    # TODO: feature selection
    # For now, pick a subset that intuitively would be predictive

    X_tr = train.iloc[:, 1:-1].to_numpy()
    y_tr = train.iloc[:, -1].to_numpy()
    X_test = test.iloc[:, 1:-1].to_numpy()
    y_test = test.iloc[:, -1].to_numpy()

    dataset = Dataset(X_tr, y_tr, 1-valid_prop, valid_prop) # Must use up the entire train set
    dataset.X_test = X_test
    dataset.y_test = y_test

    return dataset


def load_civil_war_onset(train_prop, valid_prop) -> Dataset:
    # Dataset from Ethnicity, Insurgency, and Civil War by Fearon and Laitin 2003
    # Used by https://journals.sagepub.com/doi/pdf/10.1177/2053168020905487

    raise NotImplementedError
