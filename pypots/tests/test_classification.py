"""
Test cases for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

from pypots.classification import BRITS, GRUD, Raindrop
from pypots.data import generate_random_walk_for_classification, mcar, fill_nan_with_mask
from pypots.utils.metrics import cal_binary_classification_metrics

EPOCHS = 5
N_CLASSES = 2


def gene_data():
    # generate samples
    X, y = generate_random_walk_for_classification(n_classes=N_CLASSES, n_samples_each_class=500)
    # create random missing values
    _, X, missing_mask, _ = mcar(X, 0.3)
    X = fill_nan_with_mask(X, missing_mask)
    # split into train/val/test sets
    train_X, val_X, test_X = X[:600], X[600:800], X[800:]
    train_y, val_y, test_y = y[:600], y[600:800], y[800:]
    return train_X, train_y, val_X, val_y, test_X, test_y


# def gene_physionet2012_data():
#     from pypots.data import load_specific_dataset
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import train_test_split
#     # generate samples
#     df = load_specific_dataset('physionet_2012')
#     X = df['X']
#     y = df['y']
#     all_recordID = X['RecordID'].unique()
#     train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=0.2)
#     train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=0.2)
#     train_set = X[X['RecordID'].isin(train_set_ids)]
#     val_set = X[X['RecordID'].isin(val_set_ids)]
#     test_set = X[X['RecordID'].isin(test_set_ids)]
#     train_set = train_set.drop('RecordID', axis=1)
#     val_set = val_set.drop('RecordID', axis=1)
#     test_set = test_set.drop('RecordID', axis=1)
#     train_X, val_X, test_X = train_set.to_numpy(), val_set.to_numpy(), test_set.to_numpy()
#     # normalization
#     scaler = StandardScaler()
#     train_X = scaler.fit_transform(train_X)
#     val_X = scaler.transform(val_X)
#     test_X = scaler.transform(test_X)
#     # reshape into time series samples
#     train_X = train_X.reshape(len(train_set_ids), 48, -1)
#     val_X = val_X.reshape(len(val_set_ids), 48, -1)
#     test_X = test_X.reshape(len(test_set_ids), 48, -1)
#
#     train_y = y[y.index.isin(train_set_ids)]
#     val_y = y[y.index.isin(val_set_ids)]
#     test_y = y[y.index.isin(test_set_ids)]
#     train_y, val_y, test_y = train_y.to_numpy(), val_y.to_numpy(), test_y.to_numpy()
#
#     return train_X, train_y.flatten(), val_X, val_y.flatten(), test_X, test_y.flatten()
# gene_data = gene_physionet2012_data

class TestBRITS(unittest.TestCase):
    def setUp(self) -> None:
        # generate time-series classification data
        train_X, train_y, val_X, val_y, test_X, test_y = gene_data()
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.test_X = test_X
        self.test_y = test_y
        _, seq_len, n_features = train_X.shape
        self.brits = BRITS(seq_len, n_features, 256, n_classes=N_CLASSES, epochs=EPOCHS)
        self.brits.fit(self.train_X, self.train_y, self.val_X, self.val_y)

    def test_parameters(self):
        assert (hasattr(self.brits, 'model')
                and self.brits.model is not None)

        assert (hasattr(self.brits, 'optimizer')
                and self.brits.optimizer is not None)

        assert hasattr(self.brits, 'best_loss')
        self.assertNotEqual(self.brits.best_loss, float('inf'))

        assert (hasattr(self.brits, 'best_model_dict')
                and self.brits.best_model_dict is not None)

    def test_classify(self):
        predictions = self.brits.classify(self.test_X)
        metrics = cal_binary_classification_metrics(predictions, self.test_y)
        print(metrics)
        assert metrics['roc_auc'] >= 0.5, 'ROC AUC < 0.5, there must be bugs here'


class TestGRUD(unittest.TestCase):
    def setUp(self) -> None:
        train_X, train_y, val_X, val_y, test_X, test_y = gene_data()
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.test_X = test_X
        self.test_y = test_y
        _, seq_len, n_features = train_X.shape
        self.grud = GRUD(seq_len, n_features, 256, n_classes=N_CLASSES, epochs=EPOCHS)
        self.grud.fit(self.train_X, self.train_y, self.val_X, self.val_y)

    def test_parameters(self):
        assert (hasattr(self.grud, 'model')
                and self.grud.model is not None)

        assert (hasattr(self.grud, 'optimizer')
                and self.grud.optimizer is not None)

        assert hasattr(self.grud, 'best_loss')
        self.assertNotEqual(self.grud.best_loss, float('inf'))

        assert (hasattr(self.grud, 'best_model_dict')
                and self.grud.best_model_dict is not None)

    def test_classify(self):
        predictions = self.grud.classify(self.test_X)
        metrics = cal_binary_classification_metrics(predictions, self.test_y)
        print(metrics)
        assert metrics['roc_auc'] >= 0.5, 'ROC AUC < 0.5, there must be bugs here'


class TestRaindrop(unittest.TestCase):
    def setUp(self) -> None:
        train_X, train_y, val_X, val_y, test_X, test_y = gene_data()
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.test_X = test_X
        self.test_y = test_y
        _, seq_len, n_features = train_X.shape
        self.raindrop = Raindrop(n_features, 2, 148, 256, 2, N_CLASSES, 0.3, seq_len, 0, 'mean',
                                 False, False, epochs=EPOCHS)
        self.raindrop.fit(self.train_X, self.train_y, self.val_X, self.val_y)

    def test_parameters(self):
        assert (hasattr(self.raindrop, 'model')
                and self.raindrop.model is not None)

        assert (hasattr(self.raindrop, 'optimizer')
                and self.raindrop.optimizer is not None)

        assert hasattr(self.raindrop, 'best_loss')
        self.assertNotEqual(self.raindrop.best_loss, float('inf'))

        assert (hasattr(self.raindrop, 'best_model_dict')
                and self.raindrop.best_model_dict is not None)

    def test_classify(self):
        predictions = self.raindrop.classify(self.test_X)
        metrics = cal_binary_classification_metrics(predictions, self.test_y)
        print(metrics)
        assert metrics['roc_auc'] >= 0.5, 'ROC AUC < 0.5, there must be bugs here'


if __name__ == '__main__':
    unittest.main()
