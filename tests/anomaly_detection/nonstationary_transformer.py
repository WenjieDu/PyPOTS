"""
Test cases for NonstationaryTransformer anomaly detection model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os.path
import unittest

import pytest

from pypots.anomaly_detection import NonstationaryTransformer
from pypots.nn.functional import calc_acc, calc_precision_recall_f1
from pypots.optim import Adam
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
    DEVICE,
    EPOCHS,
    GENERAL_H5_TEST_SET_PATH,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION,
    TEST_SET,
    TRAIN_SET,
    VAL_SET,
    check_tb_and_model_checkpoints_existence,
)


class TestNonstationaryTransformer(unittest.TestCase):
    logger.info("Running tests for an anomaly detection model NonstationaryTransformer...")

    # Define where to save logs and models
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION, "NonstationaryTransformer")
    model_save_name = "saved_nonstationary_transformer_model.pypots"

    # Instantiate a custom Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # Initialize the NonstationaryTransformer anomaly detection model
    nonstationary_transformer = NonstationaryTransformer(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        anomaly_rate=DATA["anomaly_rate"],
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ffn=32,
        d_projector_hidden=[64, 64],
        n_projector_hidden_layers=2,
        dropout=0,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="anomaly-detection-nonstationary_transformer")
    def test_0_fit(self):
        """Test training the model on in-memory data."""
        self.nonstationary_transformer.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="anomaly-detection-nonstationary_transformer")
    def test_1_detect(self):
        """Test anomaly detection and evaluate accuracy, precision, recall, and F1."""
        anomaly_detection_results = self.nonstationary_transformer.predict(TEST_SET)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(
            f"NonstationaryTransformer Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}"
        )

    @pytest.mark.xdist_group(name="anomaly-detection-nonstationary_transformer")
    def test_2_parameters(self):
        """Check key parameters are initialized correctly after training."""
        assert hasattr(self.nonstationary_transformer, "model") and self.nonstationary_transformer.model is not None
        assert (
            hasattr(self.nonstationary_transformer, "optimizer")
            and self.nonstationary_transformer.optimizer is not None
        )
        assert hasattr(self.nonstationary_transformer, "best_loss")
        self.assertNotEqual(self.nonstationary_transformer.best_loss, float("inf"))
        assert (
            hasattr(self.nonstationary_transformer, "best_model_dict")
            and self.nonstationary_transformer.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="anomaly-detection-nonstationary_transformer")
    def test_3_saving_path(self):
        """Test model saving and loading functionality."""
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"
        check_tb_and_model_checkpoints_existence(self.nonstationary_transformer)

        # Save model to disk and test loading
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.nonstationary_transformer.save(saved_model_path)
        self.nonstationary_transformer.load(saved_model_path)

    @pytest.mark.xdist_group(name="anomaly-detection-nonstationary_transformer")
    def test_4_lazy_loading(self):
        """Test training and prediction with lazy loading from HDF5 files."""
        self.nonstationary_transformer.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        anomaly_detection_results = self.nonstationary_transformer.predict(GENERAL_H5_TEST_SET_PATH)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(
            f"Lazy-loading NonstationaryTransformer Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}"
        )


if __name__ == "__main__":
    unittest.main()
