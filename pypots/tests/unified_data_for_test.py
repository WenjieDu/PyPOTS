"""
Generate the unified test data.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pypots.data import generate_random_walk_for_classification, mcar, masked_fill
from pypots.data import load_specific_dataset


def save_data_set_into_h5(data, path):
    with h5py.File(path, "w") as hf:
        for i in data.keys():
            hf.create_dataset(i, data=data[i].astype(np.float32))


def gene_random_walk_data(
    n_steps=24, n_features=10, n_classes=2, n_samples_each_class=1000
):
    """Generate a random-walk dataset."""
    # generate samples
    X, y = generate_random_walk_for_classification(
        n_classes=n_classes,
        n_samples_each_class=n_samples_each_class,
        n_steps=n_steps,
        n_features=n_features,
    )
    # split into train/val/test sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)
    # create random missing values
    _, train_X, missing_mask, _ = mcar(train_X, 0.3)
    train_X = masked_fill(train_X, 1 - missing_mask, torch.nan)
    _, val_X, missing_mask, _ = mcar(val_X, 0.3)
    val_X = masked_fill(val_X, 1 - missing_mask, torch.nan)
    # test set is left to mask after normalization

    train_X = train_X.reshape(-1, n_features)
    val_X = val_X.reshape(-1, n_features)
    test_X = test_X.reshape(-1, n_features)
    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    # reshape into time series samples
    train_X = train_X.reshape(-1, n_steps, n_features)
    val_X = val_X.reshape(-1, n_steps, n_features)
    test_X = test_X.reshape(-1, n_steps, n_features)

    # mask values in the test set as ground truth
    test_X_intact, test_X, test_X_missing_mask, test_X_indicating_mask = mcar(
        test_X, 0.3
    )
    test_X = masked_fill(test_X, 1 - test_X_missing_mask, torch.nan)

    data = {
        "n_classes": n_classes,
        "n_steps": n_steps,
        "n_features": n_features,
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
        "test_X_intact": test_X_intact,
        "test_X_indicating_mask": test_X_indicating_mask,
    }
    return data


def gene_physionet2012():
    """Generate PhysioNet2012."""
    # generate samples
    df = load_specific_dataset("physionet_2012")
    X = df["X"]
    y = df["y"]
    all_recordID = X["RecordID"].unique()
    train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=0.2)
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=0.2)
    train_set = X[X["RecordID"].isin(train_set_ids)]
    val_set = X[X["RecordID"].isin(val_set_ids)]
    test_set = X[X["RecordID"].isin(test_set_ids)]
    train_set = train_set.drop("RecordID", axis=1)
    val_set = val_set.drop("RecordID", axis=1)
    test_set = test_set.drop("RecordID", axis=1)
    train_X, val_X, test_X = (
        train_set.to_numpy(),
        val_set.to_numpy(),
        test_set.to_numpy(),
    )
    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    # reshape into time series samples
    train_X = train_X.reshape(len(train_set_ids), 48, -1)
    val_X = val_X.reshape(len(val_set_ids), 48, -1)
    test_X = test_X.reshape(len(test_set_ids), 48, -1)

    train_y = y[y.index.isin(train_set_ids)]
    val_y = y[y.index.isin(val_set_ids)]
    test_y = y[y.index.isin(test_set_ids)]
    train_y, val_y, test_y = train_y.to_numpy(), val_y.to_numpy(), test_y.to_numpy()

    test_X_intact, test_X, test_X_missing_mask, test_X_indicating_mask = mcar(
        test_X, 0.1
    )
    test_X = masked_fill(test_X, 1 - test_X_missing_mask, torch.nan)

    data = {
        "n_classes": 2,
        "n_steps": 48,
        "n_features": train_X.shape[-1],
        "train_X": train_X,
        "train_y": train_y.flatten(),
        "val_X": val_X,
        "val_y": val_y.flatten(),
        "test_X": test_X,
        "test_y": test_y.flatten(),
        "test_X_intact": test_X_intact,
        "test_X_indicating_mask": test_X_indicating_mask,
    }
    return data


# generate and cache data first.
# Otherwise, file lock will cause bug if running test parallely with pytest-xdist.
DATA = gene_random_walk_data()

TRAIN_SET = "./train_set.h5"
VAL_SET = "./val_set.h5"
TEST_SET = "./test_set.h5"

IMPUTATION_TRAIN_SET = "./imputation_train_set.h5"
IMPUTATION_VAL_SET = "./imputation_val_set.h5"

save_data_set_into_h5({"X": DATA["train_X"], "y": DATA["train_y"]}, TRAIN_SET)
save_data_set_into_h5({"X": DATA["val_X"], "y": DATA["val_y"]}, VAL_SET)
save_data_set_into_h5(
    {
        "X": DATA["test_X"],
        "X_intact": DATA["test_X_intact"],
        "X_indicating_mask": DATA["test_X_indicating_mask"],
    },
    TEST_SET,
)

save_data_set_into_h5({"X": DATA["train_X"]}, IMPUTATION_TRAIN_SET)
save_data_set_into_h5({"X": DATA["val_X"]}, IMPUTATION_VAL_SET)
