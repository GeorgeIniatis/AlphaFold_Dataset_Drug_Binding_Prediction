import pandas as pd
import numpy as np


def load_to_pickle(working_set, path):
    path = f"Dataset_Files/{path}.pkl"
    working_set.to_pickle(path)


def load_from_pickle(path):
    path = f"Dataset_Files/{path}.pkl"
    return pd.read_pickle(path)


def load_to_csv(working_set, path):
    path = f"Dataset_Files/{path}.csv"
    working_set.to_csv(path, index=False)


def load_from_csv(path):
    path = f"Dataset_Files/{path}.csv"
    return pd.read_csv(path)


def load_to_numpy(array, path):
    path = f"Dataset_Files/{path}.npy"
    np.save(path, array)


def load_from_numpy(path):
    path = f"Dataset_Files/{path}.npy"
    return np.load(path, allow_pickle=True)


def replace_with_nan(working_set, string_to_replace):
    working_set.replace(string_to_replace, np.NaN, inplace=True)
