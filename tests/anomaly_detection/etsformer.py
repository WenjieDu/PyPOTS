"""
Test cases for ETSformer anomaly detection model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os.path
import unittest

import pytest

from pypots.anomaly_detection import ETSformer
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


class TestETSformer(unittest.TestCase):
    logger.info("Running tests for an anomaly detection model ETSformer...")

    # Define where to save logs and models
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION, "ETSformer")
    model_save_name = "saved_etsformer_model.pypots"

    # Instantiate a custom Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # Initialize the ETSformer anomaly detection model
    etsformer = ETSformer(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        anomaly_rate=DATA["anomaly_rate"],
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_model=32,
        n_heads=2,
        d_ffn=32,
        top_k=3,
        dropout=0,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="anomaly-detection-etsformer")
    def test_0_fit(self):
        """Test training the model on in-memory data."""
        self.etsformer.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="anomaly-detection-etsformer")
    def test_1_detect(self):
        """Test anomaly detection and evaluate accuracy, precision, recall, and F1."""
        anomaly_detection_results = self.etsformer.predict(TEST_SET)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"ETSformer Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")

    @pytest.mark.xdist_group(name="anomaly-detection-etsformer")
    def test_2_parameters(self):
        """Check key parameters are initialized correctly after training."""
        assert hasattr(self.etsformer, "model") and self.etsformer.model is not None
        assert hasattr(self.etsformer, "optimizer") and self.etsformer.optimizer is not None
        assert hasattr(self.etsformer, "best_loss")
        self.assertNotEqual(self.etsformer.best_loss, float("inf"))
        assert hasattr(self.etsformer, "best_model_dict") and self.etsformer.best_model_dict is not None

    @pytest.mark.xdist_group(name="anomaly-detection-etsformer")
    def test_3_saving_path(self):
        """Test model saving and loading functionality."""
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"
        check_tb_and_model_checkpoints_existence(self.etsformer)

        # Save model to disk and test loading
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.etsformer.save(saved_model_path)
        self.etsformer.load(saved_model_path)

    @pytest.mark.xdist_group(name="anomaly-detection-etsformer")
    def test_4_lazy_loading(self):
        """Test training and prediction with lazy loading from HDF5 files."""
        self.etsformer.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        anomaly_detection_results = self.etsformer.predict(GENERAL_H5_TEST_SET_PATH)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"Lazy-loading ETSformer Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == "__main__":
    unittest.main()
