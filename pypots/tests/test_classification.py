"""
Test cases for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

from pypots.classification import BRITS, GRUD, Raindrop
from pypots.utils.metrics import cal_binary_classification_metrics
from .test_data_unified import gene_data

EPOCHS = 10


class TestBRITS(unittest.TestCase):
    def setUp(self) -> None:
        # generate time-series classification data
        data = gene_data()
        self.train_X = data['train_X']
        self.train_y = data['train_y']
        self.val_X = data['val_X']
        self.val_y = data['val_y']
        self.test_X = data['test_X']
        self.test_y = data['test_y']
        self.brits = BRITS(data['n_steps'], data['n_features'], 256,
                           n_classes=data['n_classes'], epochs=EPOCHS)
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
        data = gene_data()
        self.train_X = data['train_X']
        self.train_y = data['train_y']
        self.val_X = data['val_X']
        self.val_y = data['val_y']
        self.test_X = data['test_X']
        self.test_y = data['test_y']
        self.grud = GRUD(data['n_steps'], data['n_features'], 256, n_classes=data['n_classes'], epochs=EPOCHS)
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
        data = gene_data()
        self.train_X = data['train_X']
        self.train_y = data['train_y']
        self.val_X = data['val_X']
        self.val_y = data['val_y']
        self.test_X = data['test_X']
        self.test_y = data['test_y']
        self.raindrop = Raindrop(data['n_features'], 2, 100, 256, 2, data['n_classes'], 0.3, data['n_steps'], 0, 'mean',
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
