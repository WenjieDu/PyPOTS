"""
Test cases for clustering models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import unittest

from pypots.clustering import VaDER, CRLI
from pypots.tests.unified_data_for_test import DATA
from pypots.utils.metrics import cal_rand_index, cal_cluster_purity

EPOCHS = 5


class TestCRLI(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X = DATA['train_X']
        self.train_y = DATA['train_y']
        print('Running test cases for CRLI...')
        self.crli = CRLI(n_steps=DATA['n_steps'], n_features=DATA['n_features'], n_clusters=DATA['n_classes'],
                         n_generator_layers=2, rnn_hidden_size=128, epochs=EPOCHS)
        self.crli.fit(self.train_X)

    def test_parameters(self):
        assert (hasattr(self.crli, 'model')
                and self.crli.model is not None)

        assert (hasattr(self.crli, 'G_optimizer')
                and self.crli.G_optimizer is not None)
        assert (hasattr(self.crli, 'D_optimizer')
                and self.crli.D_optimizer is not None)

        assert hasattr(self.crli, 'best_loss')
        self.assertNotEqual(self.crli.best_loss, float('inf'))

        assert (hasattr(self.crli, 'best_model_dict')
                and self.crli.best_model_dict is not None)

    def test_cluster(self):
        clustering = self.crli.cluster(self.train_X)
        RI = cal_rand_index(clustering, self.train_y)
        CP = cal_cluster_purity(clustering, self.train_y)
        print(f'RI: {RI}\nCP: {CP}')


class TestVaDER(unittest.TestCase):
    def setUp(self) -> None:
        self.train_X = DATA['train_X']
        self.train_y = DATA['train_y']
        print('Running test cases for VaDER...')
        self.vader = VaDER(n_steps=DATA['n_steps'], n_features=DATA['n_features'], n_clusters=DATA['n_classes'],
                           rnn_hidden_size=128, d_mu_stddev=5, pretrain_epochs=10, epochs=EPOCHS)
        self.vader.fit(self.train_X)

    def test_parameters(self):
        assert (hasattr(self.vader, 'model')
                and self.vader.model is not None)

        assert (hasattr(self.vader, 'optimizer')
                and self.vader.optimizer is not None)

        assert hasattr(self.vader, 'best_loss')
        self.assertNotEqual(self.vader.best_loss, float('inf'))

        assert (hasattr(self.vader, 'best_model_dict')
                and self.vader.best_model_dict is not None)

    def test_cluster(self):
        clustering = self.vader.cluster(self.train_X)
        RI = cal_rand_index(clustering, self.train_y)
        CP = cal_cluster_purity(clustering, self.train_y)
        print(f'RI: {RI}\nCP: {CP}')


if __name__ == '__main__':
    unittest.main()
