"""
Test cases for clustering models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

from pypots.clustering.crli import CRLI
from pypots.data import generate_random_walk_for_classification, mcar, fill_nan_with_mask
from pypots.utils.metrics import cal_rand_index

EPOCHS = 5
N_CLASSES = 5


def gene_data():
    # generate samples
    X, y = generate_random_walk_for_classification(n_classes=N_CLASSES, n_samples_each_class=500)
    # create random missing values
    _, X, missing_mask, _ = mcar(X, 0.3)
    X = fill_nan_with_mask(X, missing_mask)
    return X, y


class TestCRLI(unittest.TestCase):
    def setUp(self) -> None:
        X, y = gene_data()  # generate time-series data
        self.X = X
        self.y = y
        self.crli = CRLI(seq_len=24, n_features=10, n_clusters=N_CLASSES,
                         n_generator_layers=2, rnn_hidden_size=128, epochs=EPOCHS)
        self.crli.fit(X)

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
        clustering = self.crli.cluster(self.X)
        RI = cal_rand_index(clustering, self.y)
        print(f'RI: {RI}')


if __name__ == '__main__':
    unittest.main()
