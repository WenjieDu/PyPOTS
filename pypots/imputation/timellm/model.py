"""
The implementation of TimeLLM for the partially-observed time-series imputation task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _TimeLLM
from .data import DatasetForTimeLLM
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...data.dataset import BaseDataset
from ...nn.functional.cuda import autocast
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class TimeLLM(BaseNNImputer):
    """The PyTorch implementation of the TimeLLM model.
    TimeLLM is originally proposed by Jin et al. in :cite:`jin2024timellm`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    llm_model_type :
        The type of the LLM model. It can be one of  ["LLaMA", "GPT2", "BERT"].

    n_layers :
        The number of layers in the TimeLLM model.

    patch_size :
        The length of the patch for the TimeLLM model.

    patch_stride :
        The stride for the patching process in the TimeLLM model.

    d_llm :
        The dimension of the LLM model.
        Given llm_model_type, it should be 4096 for LLaMA, 768 for GPT2 and BERT.

    d_model :
        The dimension of the model.

    d_ffn :
        The dimension of the feed-forward network.

    n_heads :
        The number of heads in each layer of TimeLLM.

    dropout :
        The dropout rate for the model.

    domain_prompt_content :
        The prompt content for the domain knowledge.

    ORT_weight :
        The weight for the ORT loss, the same as SAITS.

    MIT_weight :
        The weight for the MIT loss, the same as SAITS.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    training_loss:
        The customized loss function designed by users for training the model.
        If not given, will use the default loss as claimed in the original paper.

    validation_metric:
        The customized metric function designed by users for validating the model.
        If not given, will use the default MSE metric.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose :
        Whether to print out the training logs during the training process.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        llm_model_type: str,
        patch_size: int,
        patch_stride: int,
        d_llm: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        dropout: float,
        domain_prompt_content: str,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            enable_amp=True,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_features = n_features

        # model hype-parameters
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.d_llm = d_llm
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.llm_model_type = llm_model_type
        self.dropout = dropout
        self.domain_prompt_content = domain_prompt_content

        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # set up the model
        self.model = _TimeLLM(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            d_llm=self.d_llm,
            n_heads=self.n_heads,
            llm_model_type=self.llm_model_type,
            dropout=self.dropout,
            domain_prompt_content=self.domain_prompt_content,
            ORT_weight=self.ORT_weight,
            MIT_weight=self.MIT_weight,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _organize_content_to_save(self):
        from ...version import __version__ as pypots_version

        if isinstance(self.device, list):
            # to save a DataParallel model generically, save the model.module.state_dict()
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if "llm_model" not in k}

        all_attrs = dict({})
        all_attrs["model_state_dict"] = model_state_dict
        all_attrs["pypots_version"] = pypots_version
        return all_attrs

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForTimeLLM(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForTimeLLM(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=True)

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        self.model.eval()  # set the model to evaluation mode
        # Step 1: wrap the input data with classes Dataset and DataLoader
        test_set = BaseDataset(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        # Step 2: process the data with the model
        for idx, data in enumerate(test_loader):
            inputs = self._assemble_input_for_testing(data)
            with autocast(enabled=self.amp_enabled):
                results = self.model.forward(inputs)
            imputation_collector.append(results["imputed_data"])

        # Step 3: output collection and return
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]
