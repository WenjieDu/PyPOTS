"""
Test cases for SCINet anomaly detection model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import pytest

from pypots.anomaly_detection import SCINet
from pypots.nn.functional import calc_acc, calc_precision_recall_f1
from pypots.optim import Adam
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION,
    check_tb_and_model_checkpoints_existence,
)


class TestSCINet(unittest.TestCase):
    logger.info("Running tests for an anomaly detection model SCINet...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION, "SCINet")
    model_save_name = "saved_scinet_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize an SCINet model
    scinet = SCINet(
        DATA["n_steps"],
        DATA["n_features"],
        anomaly_rate=DATA["anomaly_rate"],
        n_stacks=2,
        n_levels=2,
        n_groups=1,
        n_decoder_layers=2,
        d_hidden=64,
        kernel_size=5,
        dropout=0,
        concat_len=0,
        pos_enc=True,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="anomaly-detection-scinet")
    def test_0_fit(self):
        self.scinet.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="anomaly-detection-scinet")
    def test_1_detect(self):
        anomaly_detection_results = self.scinet.predict(TEST_SET)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"SCINet Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")

    @pytest.mark.xdist_group(name="anomaly-detection-scinet")
    def test_2_parameters(self):
        assert hasattr(self.scinet, "model") and self.scinet.model is not None

        assert hasattr(self.scinet, "optimizer") and self.scinet.optimizer is not None

        assert hasattr(self.scinet, "best_loss")
        self.assertNotEqual(self.scinet.best_loss, float("inf"))

        assert hasattr(self.scinet, "best_model_dict") and self.scinet.best_model_dict is not None

    @pytest.mark.xdist_group(name="anomaly-detection-scinet")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.scinet)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.scinet.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.scinet.load(saved_model_path)

    @pytest.mark.xdist_group(name="anomaly-detection-scinet")
    def test_4_lazy_loading(self):
        self.scinet.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        anomaly_detection_results = self.scinet.predict(GENERAL_H5_TEST_SET_PATH)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"Lazy-loading SCINet Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == "__main__":
    unittest.main()
