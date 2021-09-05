import numpy as np


def compute_indices(n, train_prop: float, valid_prop: float):
    """ Compute the indices for train, val, test splits. """
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = np.random.permutation(n)

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    return train_indices, val_indices, test_indices
