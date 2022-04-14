"""
Test cases for imputation models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import unittest

import numpy as np
from pycorruptor import mcar, fill_nan_with_mask

from pypots.data import generate_random_walk
from pypots.imputation import (
    SAITS,
    Transformer,
    BRITS,
    LOCF,
)
from pypots.utils.metrics import cal_mae

EPOCH = 5


def gene_data():
    X = generate_random_walk()  # generate time-series data
    train_X, val_X, test_X = X[:600], X[600:800], X[800:]
    _, train_X, train_X_missing_mask, _ = mcar(train_X, 0.3)
    test_X_intact, test_X, test_X_missing_mask, test_X_indicating_mask = mcar(train_X, 0.3)
    train_X = fill_nan_with_mask(train_X, train_X_missing_mask)
    test_X = fill_nan_with_mask(test_X, test_X_missing_mask)
    return train_X, val_X, test_X, test_X_intact, test_X_indicating_mask


class TestSAITS(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X, self.val_X, self.test_X, self.test_X_intact, self.test_X_indicating_mask = gene_data()
        self.saits = SAITS(n_layers=2, d_model=256, d_inner=128, n_head=4,
                           d_k=64, d_v=64, dropout=0.1, epochs=EPOCH)
        self.saits.fit(self.train_X, self.val_X)

    def test_parameters(self):
        assert (hasattr(self.saits, 'model')
                and self.saits.model is not None)

        assert (hasattr(self.saits, 'optimizer')
                and self.saits.optimizer is not None)

        assert hasattr(self.saits, 'best_loss')
        self.assertNotEqual(self.saits.best_loss, float('inf'))

        assert (hasattr(self.saits, 'best_model_dict')
                and self.saits.best_model_dict is not None)

    def test_impute(self):
        imputed_X = self.saits.impute(self.test_X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'
        test_MAE = cal_mae(imputed_X, self.test_X_intact, self.test_X_indicating_mask)
        print(f'SAITS test_MAE: {test_MAE}')


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X, self.val_X, self.test_X, self.test_X_intact, self.test_X_indicating_mask = gene_data()
        self.transformer = Transformer(n_layers=2, d_model=256, d_inner=128, n_head=4,
                                       d_k=64, d_v=64, dropout=0.1, epochs=EPOCH)
        self.transformer.fit(self.train_X, self.val_X)

    def test_parameters(self):
        assert (hasattr(self.transformer, 'model')
                and self.transformer.model is not None)

        assert (hasattr(self.transformer, 'optimizer')
                and self.transformer.optimizer is not None)

        assert hasattr(self.transformer, 'best_loss')
        self.assertNotEqual(self.transformer.best_loss, float('inf'))

        assert (hasattr(self.transformer, 'best_model_dict')
                and self.transformer.best_model_dict is not None)

    def test_impute(self):
        imputed_X = self.transformer.impute(self.test_X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'
        test_MAE = cal_mae(imputed_X, self.test_X_intact, self.test_X_indicating_mask)
        print(f'Transformer test_MAE: {test_MAE}')


class TestBRITS(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X, self.val_X, self.test_X, self.test_X_intact, self.test_X_indicating_mask = gene_data()
        self.brits = BRITS(256, epochs=EPOCH)
        self.brits.fit(self.train_X, self.val_X)

    def test_parameters(self):
        assert (hasattr(self.brits, 'model')
                and self.brits.model is not None)

        assert (hasattr(self.brits, 'optimizer')
                and self.brits.optimizer is not None)

        assert hasattr(self.brits, 'best_loss')
        self.assertNotEqual(self.brits.best_loss, float('inf'))

        assert (hasattr(self.brits, 'best_model_dict')
                and self.brits.best_model_dict is not None)

    def test_impute(self):
        imputed_X = self.brits.impute(self.test_X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'
        test_MAE = cal_mae(imputed_X, self.test_X_intact, self.test_X_indicating_mask)
        print(f'BRITS test_MAE: {test_MAE}')


class TestLOCF(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X, self.val_X, self.test_X, self.test_X_intact, self.test_X_indicating_mask = gene_data()
        self.locf = LOCF(nan=0)

    def test_parameters(self):
        assert (hasattr(self.locf, 'nan')
                and self.locf.nan is not None)

    def test_impute(self):
        imputed_X = self.locf.impute(self.test_X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'
        test_MAE = cal_mae(imputed_X, self.test_X_intact, self.test_X_indicating_mask)
        print(f'LOCF test_MAE: {test_MAE}')


if __name__ == '__main__':
    unittest.main()
