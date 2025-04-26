"""
Test cases for TimeMixerPP anomaly detection model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import pytest

from pypots.anomaly_detection import TimeMixerPP
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


class TestTimeMixerPP(unittest.TestCase):
    logger.info("Running tests for an anomaly detection model TimeMixerPP...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION, "TimeMixerPP")
    model_save_name = "saved_timemixerpp_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize an TimeMixerPP model
    timemixerpp = TimeMixerPP(
        DATA["n_steps"],
        DATA["n_features"],
        anomaly_rate=DATA["anomaly_rate"],
        n_layers=1,
        top_k=5,
        d_model=16,
        d_ffn=16,
        n_heads=4,
        n_kernels=3,
        dropout=0.1,
        downsampling_window=1,
        downsampling_layers=2,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="anomaly-detection-timemixerpp")
    def test_0_fit(self):
        self.timemixerpp.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="anomaly-detection-timemixerpp")
    def test_1_detect(self):
        anomaly_detection_results = self.timemixerpp.predict(TEST_SET)
        anomaly_labels = TEST_SET["anomaly_y"].flatten()

        accuracy = calc_acc(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        precision, recall, f1 = calc_precision_recall_f1(
            anomaly_detection_results["anomaly_detection"],
            anomaly_labels,
        )
        logger.info(f"TimeMixerPP Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")

    @pytest.mark.xdist_group(name="anomaly-detection-timemixerpp")
    def test_2_parameters(self):
        assert hasattr(self.timemixerpp, "model") and self.timemixerpp.model is not None

        assert hasattr(self.timemixerpp, "optimizer") and self.timemixerpp.optimizer is not None

        assert hasattr(self.timemixerpp, "best_loss")
        self.assertNotEqual(self.timemixerpp.best_loss, float("inf"))

        assert hasattr(self.timemixerpp, "best_model_dict") and self.timemixerpp.best_model_dict is not None

    @pytest.mark.xdist_group(name="anomaly-detection-timemixerpp")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.timemixerpp)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.timemixerpp.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.timemixerpp.load(saved_model_path)

    @pytest.mark.xdist_group(name="anomaly-detection-timemixerpp")
    def test_4_lazy_loading(self):
        self.timemixerpp.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        anomaly_detection_results = self.timemixerpp.predict(GENERAL_H5_TEST_SET_PATH)
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
            f"Lazy-loading TimeMixerPP Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}"
        )


if __name__ == "__main__":
    unittest.main()
