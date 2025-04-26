"""
Test cases for DLinear anomaly detection model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import pytest

from pypots.anomaly_detection import DLinear
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


class TestDLinear(unittest.TestCase):
    logger.info("Running tests for an anomaly detection model DLinear...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION, "DLinear")
    model_save_name = "saved_dlinear_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize an DLinear model
    dlinear = DLinear(
        DATA["n_steps"],
        DATA["n_features"],
        anomaly_rate=DATA["anomaly_rate"],
        moving_avg_window_size=3,
        individual=False,
        d_model=128,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    individual_optimizer = Adam(lr=0.001, weight_decay=1e-5)
    individual_dlinear = DLinear(
        DATA["n_steps"],
        DATA["n_features"],
        anomaly_rate=DATA["anomaly_rate"],
        moving_avg_window_size=3,
        individual=True,
        d_model=None,  # d_model is useless for DLinear in the individual mode
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=individual_optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="anomaly-detection-dlinear")
    def test_0_fit(self):
        self.dlinear.fit(TRAIN_SET, VAL_SET)
        self.individual_dlinear.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="anomaly-detection-dlinear")
    def test_1_detect(self):
        anomaly_detection_results = self.dlinear.predict(TEST_SET)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"DLinear Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")

        anomaly_detection_results = self.individual_dlinear.predict(TEST_SET)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"Individual DLinear Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")

    @pytest.mark.xdist_group(name="anomaly-detection-dlinear")
    def test_2_parameters(self):
        assert hasattr(self.dlinear, "model") and self.dlinear.model is not None

        assert hasattr(self.dlinear, "optimizer") and self.dlinear.optimizer is not None

        assert hasattr(self.dlinear, "best_loss")
        self.assertNotEqual(self.dlinear.best_loss, float("inf"))

        assert hasattr(self.dlinear, "best_model_dict") and self.dlinear.best_model_dict is not None

    @pytest.mark.xdist_group(name="anomaly-detection-dlinear")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.dlinear)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.dlinear.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.dlinear.load(saved_model_path)

    @pytest.mark.xdist_group(name="anomaly-detection-dlinear")
    def test_4_lazy_loading(self):
        self.dlinear.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        anomaly_detection_results = self.dlinear.predict(GENERAL_H5_TEST_SET_PATH)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"Lazy-loading DLinear Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == "__main__":
    unittest.main()
