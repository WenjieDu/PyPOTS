"""
Test cases for imputation models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import unittest

import numpy as np

from pypots.imputation import (
    SAITS,
    Transformer,
    BRITS,
    LOCF,
)
from pypots.utils.metrics import cal_mae
from .unified_data_for_test import gene_data

EPOCH = 5


class TestSAITS(unittest.TestCase):
    def setUp(self) -> None:
        data = gene_data()
        self.train_X = data['train_X']
        self.val_X = data['val_X']
        self.test_X = data['test_X']
        self.test_X_intact = data['test_X_intact']
        self.test_X_indicating_mask = data['test_X_indicating_mask']
        print('Running test cases for SAITS...')
        self.saits = SAITS(data['n_steps'], data['n_features'], n_layers=2, d_model=256, d_inner=128, n_head=4,
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
        data = gene_data()
        self.train_X = data['train_X']
        self.val_X = data['val_X']
        self.test_X = data['test_X']
        self.test_X_intact = data['test_X_intact']
        self.test_X_indicating_mask = data['test_X_indicating_mask']
        print('Running test cases for Transformer...')
        self.transformer = Transformer(data['n_steps'], data['n_features'], n_layers=2, d_model=256, d_inner=128,
                                       n_head=4,
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
        data = gene_data()
        self.train_X = data['train_X']
        self.val_X = data['val_X']
        self.test_X = data['test_X']
        self.test_X_intact = data['test_X_intact']
        self.test_X_indicating_mask = data['test_X_indicating_mask']
        print('Running test cases for BRITS...')
        self.brits = BRITS(data['n_steps'], data['n_features'], 256, epochs=EPOCH)
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
        data = gene_data()
        self.train_X = data['train_X']
        self.val_X = data['val_X']
        self.test_X = data['test_X']
        self.test_X_intact = data['test_X_intact']
        self.test_X_indicating_mask = data['test_X_indicating_mask']
        print('Running test cases for LOCF...')
        self.locf = LOCF(nan=0)

    def test_parameters(self):
        assert (hasattr(self.locf, 'nan')
                and self.locf.nan is not None)

    def test_impute(self):
        test_X_imputed = self.locf.impute(self.test_X)
        assert not np.isnan(test_X_imputed).any(), 'Output still has missing values after running impute().'
        test_MAE = cal_mae(test_X_imputed, self.test_X_intact, self.test_X_indicating_mask)
        print(f'LOCF test_MAE: {test_MAE}')


if __name__ == '__main__':
    unittest.main()
