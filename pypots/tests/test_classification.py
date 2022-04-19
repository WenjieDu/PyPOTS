"""
Test cases for 
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

from pypots.classification import BRITS, GRUD
from pypots.data import generate_random_walk_for_classification
from pypots.data import mcar, fill_nan_with_mask
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
        self.brits = BRITS(256, n_classes=N_CLASSES, epochs=EPOCHS)
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
        self.grud = GRUD(256, n_classes=N_CLASSES, epochs=EPOCHS)
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


if __name__ == '__main__':
    unittest.main()
