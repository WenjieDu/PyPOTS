"""
Test cases for USAD anomaly detection model.
"""

# Created by omimajleta
# License: BSD-3-Clause

import unittest

import numpy as np

from pypots.anomaly_detection.usad import USAD


class TestUSAD(unittest.TestCase):
    """Test suite for the USAD anomaly detection model."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test data for all test cases."""
        cls.n_samples = 100
        cls.n_steps = 24
        cls.n_features = 5
        cls.anomaly_rate = 0.1

        # Generate normal data
        np.random.seed(42)
        X = np.random.randn(cls.n_samples, cls.n_steps, cls.n_features)

        # Inject anomalies
        anomaly_indices = np.random.choice(
            cls.n_samples,
            size=int(cls.n_samples * cls.anomaly_rate),
            replace=False,
        )
        X[anomaly_indices] *= 10

        # Add missing values
        mask = np.random.rand(*X.shape) < 0.2
        X[mask] = np.nan

        cls.train_set = {"X": X}
        cls.test_set = {"X": X}

        cls.model = USAD(
            n_steps=cls.n_steps,
            n_features=cls.n_features,
            anomaly_rate=cls.anomaly_rate,
            d_model=32,
            epochs=5,
            batch_size=32,
        )

    def test_01_fit(self):
        """Test that the model trains without errors."""
        self.model.fit(self.train_set)

    def test_02_predict(self):
        """Test that predict returns correct keys and shapes."""
        result = self.model.predict(self.test_set)

        self.assertIn("anomaly_scores", result)
        self.assertIn("anomaly_labels", result)
        self.assertEqual(result["anomaly_scores"].shape[0], self.n_samples)
        self.assertEqual(result["anomaly_labels"].shape[0], self.n_samples)

    def test_03_anomaly_labels_binary(self):
        """Test that anomaly labels are binary (0 or 1)."""
        result = self.model.predict(self.test_set)
        labels = result["anomaly_labels"]
        self.assertTrue(
            np.all((labels == 0) | (labels == 1)),
            "Anomaly labels must be binary (0 or 1)",
        )

    def test_04_anomaly_rate(self):
        """Test that detected anomaly rate is close to the configured rate."""
        result = self.model.predict(self.test_set)
        detected_rate = result["anomaly_labels"].mean()
        self.assertAlmostEqual(
            detected_rate,
            self.anomaly_rate,
            delta=0.05,
            msg="Detected anomaly rate should be close to configured anomaly_rate",
        )

    def test_05_invalid_train_set_type(self):
        """Test that a non-dict train_set raises TypeError."""
        with self.assertRaises(TypeError):
            self.model.fit([1, 2, 3])

    def test_06_missing_x_key(self):
        """Test that a train_set without 'X' raises KeyError."""
        with self.assertRaises(KeyError):
            self.model.fit({"data": np.zeros((10, 24, 5))})


if __name__ == "__main__":
    unittest.main()
