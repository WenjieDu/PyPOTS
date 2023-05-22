"""
Test cases for running models on multi cuda devices.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import os.path
import unittest

import numpy as np
import pytest
import torch

from pypots.classification import BRITS, GRUD, Raindrop
from pypots.clustering import VaDER, CRLI
from pypots.forecasting import BTTF
from pypots.imputation import BRITS as ImputationBRITS
from pypots.imputation import (
    SAITS,
    Transformer,
    MRNN,
    LOCF,
)
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_binary_classification_metrics
from pypots.utils.metrics import cal_mae
from pypots.utils.metrics import cal_rand_index, cal_cluster_purity
from tests.global_test_config import (
    DATA,
    RESULT_SAVING_DIR,
    check_tb_and_model_checkpoints_existence,
)

EPOCHS = 5

DEVICES = [torch.device(i) for i in range(torch.cuda.device_count())]
LESS_THAN_TWO_DEVICES = len(DEVICES) < 2

# global skip test if less than two cuda-enabled devices
pytestmark = pytest.mark.skipif(LESS_THAN_TWO_DEVICES, reason="not enough cuda devices")


TRAIN_SET = {"X": DATA["train_X"], "y": DATA["train_y"]}

VAL_SET = {
    "X": DATA["val_X"],
    "X_intact": DATA["val_X_intact"],
    "indicating_mask": DATA["val_X_indicating_mask"],
    "y": DATA["val_y"],
}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_IMPUTATION = os.path.join(RESULT_SAVING_DIR, "imputation")
RESULT_SAVING_DIR_FOR_CLASSIFICATION = os.path.join(RESULT_SAVING_DIR, "classification")
RESULT_SAVING_DIR_FOR_CLUSTERING = os.path.join(RESULT_SAVING_DIR, "clustering")


class TestSAITS(unittest.TestCase):
    logger.info("Running tests for an imputation model SAITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "SAITS")
    model_save_name = "saved_saits_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICES,
    )

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_1_impute(self):
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_2_parameters(self):
        assert hasattr(self.saits, "model") and self.saits.model is not None

        assert hasattr(self.saits, "optimizer") and self.saits.optimizer is not None

        assert hasattr(self.saits, "best_loss")
        self.assertNotEqual(self.saits.best_loss, float("inf"))

        assert (
            hasattr(self.saits, "best_model_dict")
            and self.saits.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.saits)

        # save the trained model into file, and check if the path exists
        self.saits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.saits.load_model(saved_model_path)


class TestTransformer(unittest.TestCase):
    logger.info("Running tests for an imputation model Transformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "Transformer")
    model_save_name = "saved_transformer_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a Transformer model
    transformer = Transformer(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICES,
    )

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_0_fit(self):
        self.transformer.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_1_impute(self):
        imputed_X = self.transformer.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"Transformer test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_2_parameters(self):
        assert hasattr(self.transformer, "model") and self.transformer.model is not None

        assert (
            hasattr(self.transformer, "optimizer")
            and self.transformer.optimizer is not None
        )

        assert hasattr(self.transformer, "best_loss")
        self.assertNotEqual(self.transformer.best_loss, float("inf"))

        assert (
            hasattr(self.transformer, "best_model_dict")
            and self.transformer.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.transformer)

        # save the trained model into file, and check if the path exists
        self.transformer.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.transformer.load_model(saved_model_path)


class TestImputationBRITS(unittest.TestCase):
    logger.info("Running tests for an imputation model BRITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "BRITS")
    model_save_name = "saved_BRITS_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a BRITS model
    brits = ImputationBRITS(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        epochs=EPOCHS,
        saving_path=f"{RESULT_SAVING_DIR_FOR_IMPUTATION}/BRITS",
        optimizer=optimizer,
        device=DEVICES,
    )

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_0_fit(self):
        self.brits.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_1_impute(self):
        imputed_X = self.brits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"BRITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_2_parameters(self):
        assert hasattr(self.brits, "model") and self.brits.model is not None

        assert hasattr(self.brits, "optimizer") and self.brits.optimizer is not None

        assert hasattr(self.brits, "best_loss")
        self.assertNotEqual(self.brits.best_loss, float("inf"))

        assert (
            hasattr(self.brits, "best_model_dict")
            and self.brits.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.brits)

        # save the trained model into file, and check if the path exists
        self.brits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.brits.load_model(saved_model_path)


class TestMRNN(unittest.TestCase):
    logger.info("Running tests for an imputation model MRNN...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "MRNN")
    model_save_name = "saved_MRNN_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a MRNN model
    mrnn = MRNN(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        epochs=EPOCHS,
        saving_path=f"{RESULT_SAVING_DIR_FOR_IMPUTATION}/MRNN",
        optimizer=optimizer,
        device=DEVICES,
    )

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_0_fit(self):
        self.mrnn.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_1_impute(self):
        imputed_X = self.mrnn.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"MRNN test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_2_parameters(self):
        assert hasattr(self.mrnn, "model") and self.mrnn.model is not None

        assert hasattr(self.mrnn, "optimizer") and self.mrnn.optimizer is not None

        assert hasattr(self.mrnn, "best_loss")
        self.assertNotEqual(self.mrnn.best_loss, float("inf"))

        assert (
            hasattr(self.mrnn, "best_model_dict")
            and self.mrnn.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.mrnn)

        # save the trained model into file, and check if the path exists
        self.mrnn.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.mrnn.load_model(saved_model_path)


class TestLOCF(unittest.TestCase):
    logger.info("Running tests for an imputation model LOCF...")
    locf = LOCF(nan=0)

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_0_impute(self):
        test_X_imputed = self.locf.impute(TEST_SET)
        assert not np.isnan(
            test_X_imputed
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"LOCF test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_1_parameters(self):
        assert hasattr(self.locf, "nan") and self.locf.nan is not None


class TestClassificationBRITS(unittest.TestCase):
    logger.info("Running tests for a classification model BRITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "BRITS")
    model_save_name = "saved_BRITS_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a BRITS model
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        rnn_hidden_size=256,
        epochs=EPOCHS,
        saving_path=saving_path,
        model_saving_strategy="better",
        optimizer=optimizer,
        device=DEVICES,
    )

    @pytest.mark.xdist_group(name="classification-brits")
    def test_0_fit(self):
        self.brits.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-brits")
    def test_1_classify(self):
        predictions = self.brits.classify(TEST_SET)
        metrics = cal_binary_classification_metrics(predictions, DATA["test_y"])
        logger.info(
            f'ROC_AUC: {metrics["roc_auc"]}, \n'
            f'PR_AUC: {metrics["pr_auc"]},\n'
            f'F1: {metrics["f1"]},\n'
            f'Precision: {metrics["precision"]},\n'
            f'Recall: {metrics["recall"]},\n'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-brits")
    def test_2_parameters(self):
        assert hasattr(self.brits, "model") and self.brits.model is not None

        assert hasattr(self.brits, "optimizer") and self.brits.optimizer is not None

        assert hasattr(self.brits, "best_loss")
        self.assertNotEqual(self.brits.best_loss, float("inf"))

        assert (
            hasattr(self.brits, "best_model_dict")
            and self.brits.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="classification-brits")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.brits)

        # save the trained model into file, and check if the path exists
        self.brits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.brits.load_model(saved_model_path)


class TestGRUD(unittest.TestCase):
    logger.info("Running tests for a classification model GRUD...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "GRUD")
    model_save_name = "saved_GRUD_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a GRUD model
    grud = GRUD(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        rnn_hidden_size=256,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICES,
    )

    @pytest.mark.xdist_group(name="classification-grud")
    def test_0_fit(self):
        self.grud.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-grud")
    def test_1_classify(self):
        predictions = self.grud.classify(TEST_SET)
        metrics = cal_binary_classification_metrics(predictions, DATA["test_y"])
        logger.info(
            f'ROC_AUC: {metrics["roc_auc"]}, \n'
            f'PR_AUC: {metrics["pr_auc"]},\n'
            f'F1: {metrics["f1"]},\n'
            f'Precision: {metrics["precision"]},\n'
            f'Recall: {metrics["recall"]},\n'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-grud")
    def test_2_parameters(self):
        assert hasattr(self.grud, "model") and self.grud.model is not None

        assert hasattr(self.grud, "optimizer") and self.grud.optimizer is not None

        assert hasattr(self.grud, "best_loss")
        self.assertNotEqual(self.grud.best_loss, float("inf"))

        assert (
            hasattr(self.grud, "best_model_dict")
            and self.grud.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="classification-grud")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.grud)

        # save the trained model into file, and check if the path exists
        self.grud.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.grud.load_model(saved_model_path)


class TestRaindrop(unittest.TestCase):
    logger.info("Running tests for a classification model Raindrop...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "Raindrop")
    model_save_name = "saved_Raindrop_model.pypots"

    # initialize a Raindrop model
    raindrop = Raindrop(
        DATA["n_steps"],
        DATA["n_features"],
        DATA["n_classes"],
        n_layers=2,
        d_model=DATA["n_features"] * 4,
        d_inner=256,
        n_heads=2,
        dropout=0.3,
        d_static=0,
        aggregation="mean",
        sensor_wise_mask=False,
        static=False,
        epochs=EPOCHS,
        saving_path=saving_path,
    )

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_0_fit(self):
        self.raindrop.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_1_classify(self):
        predictions = self.raindrop.classify(TEST_SET)
        metrics = cal_binary_classification_metrics(predictions, DATA["test_y"])
        logger.info(
            f'ROC_AUC: {metrics["roc_auc"]}, \n'
            f'PR_AUC: {metrics["pr_auc"]},\n'
            f'F1: {metrics["f1"]},\n'
            f'Precision: {metrics["precision"]},\n'
            f'Recall: {metrics["recall"]},\n'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_2_parameters(self):
        assert hasattr(self.raindrop, "model") and self.raindrop.model is not None

        assert (
            hasattr(self.raindrop, "optimizer") and self.raindrop.optimizer is not None
        )

        assert hasattr(self.raindrop, "best_loss")
        self.assertNotEqual(self.raindrop.best_loss, float("inf"))

        assert (
            hasattr(self.raindrop, "best_model_dict")
            and self.raindrop.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.raindrop)

        # save the trained model into file, and check if the path exists
        self.raindrop.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.raindrop.load_model(saved_model_path)


class TestCRLI(unittest.TestCase):
    logger.info("Running tests for a clustering model CRLI...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLUSTERING, "CRLI")
    model_save_name = "saved_CRLI_model.pypots"

    # initialize an Adam optimizer
    G_optimizer = Adam(lr=0.001, weight_decay=1e-5)
    D_optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a CRLI model
    crli = CRLI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        n_generator_layers=2,
        rnn_hidden_size=128,
        epochs=EPOCHS,
        saving_path=saving_path,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
    )

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_0_fit(self):
        self.crli.fit(TRAIN_SET)

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_1_parameters(self):
        assert hasattr(self.crli, "model") and self.crli.model is not None

        assert hasattr(self.crli, "G_optimizer") and self.crli.G_optimizer is not None
        assert hasattr(self.crli, "D_optimizer") and self.crli.D_optimizer is not None

        assert hasattr(self.crli, "best_loss")
        self.assertNotEqual(self.crli.best_loss, float("inf"))

        assert (
            hasattr(self.crli, "best_model_dict")
            and self.crli.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_2_cluster(self):
        clustering = self.crli.cluster(TEST_SET)
        RI = cal_rand_index(clustering, DATA["test_y"])
        CP = cal_cluster_purity(clustering, DATA["test_y"])
        logger.info(f"RI: {RI}\nCP: {CP}")

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.crli)

        # save the trained model into file, and check if the path exists
        self.crli.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.crli.load_model(saved_model_path)


class TestVaDER(unittest.TestCase):
    logger.info("Running tests for a clustering model Transformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLUSTERING, "VaDER")
    model_save_name = "saved_VaDER_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a VaDER model
    vader = VaDER(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        rnn_hidden_size=64,
        d_mu_stddev=5,
        pretrain_epochs=20,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICES,
    )

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_0_fit(self):
        self.vader.fit(TRAIN_SET)

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_1_cluster(self):
        try:
            clustering = self.vader.cluster(TEST_SET)
            RI = cal_rand_index(clustering, DATA["test_y"])
            CP = cal_cluster_purity(clustering, DATA["test_y"])
            logger.info(f"RI: {RI}\nCP: {CP}")
        except np.linalg.LinAlgError as e:
            logger.error(
                f"{e}\n"
                "Got singular matrix, please try to retrain the model to fix this"
            )

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_2_parameters(self):
        assert hasattr(self.vader, "model") and self.vader.model is not None

        assert hasattr(self.vader, "optimizer") and self.vader.optimizer is not None

        assert hasattr(self.vader, "best_loss")
        self.assertNotEqual(self.vader.best_loss, float("inf"))

        assert (
            hasattr(self.vader, "best_model_dict")
            and self.vader.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.vader)

        # save the trained model into file, and check if the path exists
        self.vader.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.vader.load_model(saved_model_path)


class TestBTTF(unittest.TestCase):
    logger.info("Running tests for a forecasting model BTTF...")

    # initialize a BTTF model
    bttf = BTTF(
        n_steps=50,
        n_features=10,
        pred_step=10,
        rank=10,
        time_lags=[1, 2, 3, 10, 10 + 1, 10 + 2, 20, 20 + 1, 20 + 2],
        burn_iter=5,
        gibbs_iter=5,
        multi_step=1,
    )

    @pytest.mark.xdist_group(name="forecasting-bttf")
    def test_0_forecasting(self):
        predictions = self.bttf.forecast(TEST_SET)
        logger.info(f"prediction shape: {predictions.shape}")
        mae = cal_mae(predictions, DATA["test_X_intact"][:, 50:])
        logger.info(f"prediction MAE: {mae}")


if __name__ == "__main__":
    unittest.main()
