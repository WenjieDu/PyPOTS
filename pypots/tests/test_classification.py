"""
Test cases for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

from pypots.classification import BRITS, GRUD, Raindrop
from pypots.tests.unified_data_for_test import DATA
from pypots.utils.metrics import cal_binary_classification_metrics

EPOCHS = 5


class TestBRITS(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X = DATA['train_X']
        self.train_y = DATA['train_y']
        self.val_X = DATA['val_X']
        self.val_y = DATA['val_y']
        self.test_X = DATA['test_X']
        self.test_y = DATA['test_y']
        print('Running test cases for BRITS...')
        self.brits = BRITS(DATA['n_steps'], DATA['n_features'], 256,
                           n_classes=DATA['n_classes'], epochs=EPOCHS)
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
        print(f'ROC_AUC: {metrics["roc_auc"]}, \n'
              f'PR_AUC: {metrics["pr_auc"]},\n'
              f'F1: {metrics["f1"]},\n'
              f'Precision: {metrics["precision"]},\n'
              f'Recall: {metrics["recall"]},\n')
        assert metrics['roc_auc'] >= 0.5, 'ROC-AUC < 0.5'


class TestGRUD(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X = DATA['train_X']
        self.train_y = DATA['train_y']
        self.val_X = DATA['val_X']
        self.val_y = DATA['val_y']
        self.test_X = DATA['test_X']
        self.test_y = DATA['test_y']
        print('Running test cases for GRUD...')
        self.grud = GRUD(DATA['n_steps'], DATA['n_features'], 256, n_classes=DATA['n_classes'], epochs=EPOCHS)
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
        print(f'ROC_AUC: {metrics["roc_auc"]}, \n'
              f'PR_AUC: {metrics["pr_auc"]},\n'
              f'F1: {metrics["f1"]},\n'
              f'Precision: {metrics["precision"]},\n'
              f'Recall: {metrics["recall"]},\n')
        assert metrics['roc_auc'] >= 0.5, 'ROC-AUC < 0.5'


class TestRaindrop(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X = DATA['train_X']
        self.train_y = DATA['train_y']
        self.val_X = DATA['val_X']
        self.val_y = DATA['val_y']
        self.test_X = DATA['test_X']
        self.test_y = DATA['test_y']
        print('Running test cases for Raindrop...')
        self.raindrop = Raindrop(DATA['n_features'], 2, DATA['n_features'] * 4, 256, 2, DATA['n_classes'], 0.3,
                                 DATA['n_steps'], 0, 'mean', False, False, epochs=EPOCHS)
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
        print(f'ROC_AUC: {metrics["roc_auc"]}, \n'
              f'PR_AUC: {metrics["pr_auc"]},\n'
              f'F1: {metrics["f1"]},\n'
              f'Precision: {metrics["precision"]},\n'
              f'Recall: {metrics["recall"]},\n')
        assert metrics['roc_auc'] >= 0.5, 'ROC-AUC < 0.5'


if __name__ == '__main__':
    unittest.main()
