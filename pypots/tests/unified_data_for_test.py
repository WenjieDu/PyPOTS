"""
Generate the unified test data.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pypots.data import generate_random_walk_for_classification, mcar, fill_nan_with_mask
from pypots.data import load_specific_dataset


def gene_random_walk_data():
    """ Generate a random-walk dataset.
    """
    n_classes = 2
    n_samples_each_class = 1000
    n_steps = 24
    n_features = 10

    # generate samples
    X, y = generate_random_walk_for_classification(n_classes=n_classes,
                                                   n_samples_each_class=n_samples_each_class,
                                                   n_steps=n_steps,
                                                   n_features=n_features)
    # split into train/val/test sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)
    # create random missing values
    _, train_X, missing_mask, _ = mcar(train_X, 0.3)
    train_X = fill_nan_with_mask(train_X, missing_mask)
    _, val_X, missing_mask, _ = mcar(val_X, 0.3)
    val_X = fill_nan_with_mask(val_X, missing_mask)
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
    test_X_intact, test_X, test_X_missing_mask, test_X_indicating_mask = mcar(test_X, 0.3)
    test_X = fill_nan_with_mask(test_X, test_X_missing_mask)

    data = {
        'n_classes': n_classes,
        'n_steps': n_steps,
        'n_features': n_features,
        'train_X': train_X, 'train_y': train_y,
        'val_X': val_X, 'val_y': val_y,
        'test_X': test_X, 'test_y': test_y,
        'test_X_intact': test_X_intact,
        'test_X_indicating_mask': test_X_indicating_mask
    }
    return data


def gene_physionet2012():
    """ Generate PhysioNet2012.
    """
    # generate samples
    df = load_specific_dataset('physionet_2012')
    X = df['X']
    X = X.drop(df['static_features'], axis=1)

    def apply_func(df_temp):
        missing = list(set(range(0, 48)).difference(set(df_temp['Time'])))
        missing_part = pd.DataFrame({'Time': missing})
        df_temp = df_temp.append(missing_part, ignore_index=False, sort=False)
        df_temp = df_temp.set_index('Time').sort_index().reset_index()
        df_temp = df_temp.iloc[:48]
        return df_temp

    X = X.groupby('RecordID').apply(apply_func)
    X = X.drop('RecordID', axis=1)
    X = X.reset_index()
    X = X.drop(['level_1', 'Time'], axis=1)

    y = df['y']
    all_recordID = X['RecordID'].unique()
    train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=0.2)
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=0.2)
    train_set = X[X['RecordID'].isin(train_set_ids)]
    val_set = X[X['RecordID'].isin(val_set_ids)]
    test_set = X[X['RecordID'].isin(test_set_ids)]
    train_set = train_set.drop('RecordID', axis=1)
    val_set = val_set.drop('RecordID', axis=1)
    test_set = test_set.drop('RecordID', axis=1)
    train_X, val_X, test_X = train_set.to_numpy(), val_set.to_numpy(), test_set.to_numpy()
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

    test_X_intact, test_X, test_X_missing_mask, test_X_indicating_mask = mcar(test_X, 0.1)
    test_X = fill_nan_with_mask(test_X, test_X_missing_mask)

    data = {
        'n_classes': 2,
        'n_steps': 48,
        'n_features': train_X.shape[-1],
        'train_X': train_X, 'train_y': train_y.flatten(),
        'val_X': val_X, 'val_y': val_y.flatten(),
        'test_X': test_X, 'test_y': test_y.flatten(),
        'test_X_intact': test_X_intact,
        'test_X_indicating_mask': test_X_indicating_mask
    }
    return data


# generate and cache data first.
# Otherwise, file lock will cause bug if running test parallely with pytest-xdist.
DATA = gene_random_walk_data()
