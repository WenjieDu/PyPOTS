"""Standalone LIFTS joint-training and checkpoint policy (lambda_enc=0).

Ported from the official FBIImputer.fit default joint-training path at
dd03fd1ae1cf426251dc434c4dd3ba323f2e1d40. Copyright LIFTS contributors;
MIT license, see LICENSE. Validation never chooses the checkpoint here.
"""

from copy import deepcopy

import numpy as np

from ....utils.logging import logger


def reference_parameter_order(backbone):
    """Use the exact parameter ordering of the original single Adam optimizer."""
    return [
        parameter
        for name in (
            "transition_g",
            "out_fx",
            "out_fo",
            "noise_processor",
            "encoder_h",
            "backward_transition_q",
            "decoder_u",
        )
        for parameter in getattr(backbone, name).parameters()
    ]


class ReferenceTrainingMixin:
    """Keep PyPOTS data/API contracts and the original LIFTS training policy."""

    def _train_model(self, train_dataloader, val_dataloader=None):
        if val_dataloader is not None:
            logger.info(
                "LIFTS follows the original training-loss checkpoint policy; "
                "validation data do not select checkpoints or consume sampling RNG."
            )
        if self.original_patience >= 0:
            logger.info("LIFTS follows the original full-epoch schedule; patience does not stop training.")
        self.best_model_dict = None
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.training_history_ = []
        step = 0
        for epoch in range(self.epochs):
            self.model.train()
            totals, forecasts, imputations = [], [], []
            for data in train_dataloader:
                inputs = self._assemble_input_for_training(data)
                self.optimizer.zero_grad()
                results = self.model(inputs, calc_criterion=True)
                loss = results["loss"]
                loss.backward()
                self.optimizer.step()
                totals.append(loss.item())
                forecasts.append(results["forecast_loss"].item())
                imputations.append(results["self_mask_loss"].item())
                step += 1
                if self.summary_writer is not None:
                    self._save_log_into_tb_file(step, "training", dict(results))
            mean_loss = float(np.mean(totals))
            if not np.isfinite(mean_loss):
                raise ValueError("LIFTS joint training produced a non-finite loss.")
            # Exact source policy: zero-based epoch >= total_loops // 4.
            improved = epoch >= self.epochs // 4 and mean_loss < self.best_loss
            if improved:
                self.best_loss = mean_loss
                self.best_epoch = epoch + 1
                self.best_model_dict = deepcopy(self.model.state_dict())
            self.training_history_.append(
                {
                    "epoch": epoch + 1,
                    "joint_loss": mean_loss,
                    "forecast_loss": float(np.mean(forecasts)),
                    "self_mask_loss": float(np.mean(imputations)),
                    "checkpoint_eligible": epoch >= self.epochs // 4,
                    "selected": improved,
                }
            )
            if self.verbose:
                logger.info(
                    f"LIFTS epoch {epoch + 1:03d}/{self.epochs}: joint loss {mean_loss:.6f}; "
                    f"best epoch {self.best_epoch}"
                )
            self._auto_save_model_if_necessary(
                confirm_saving=improved and self.model_saving_strategy == "better",
                saving_name=f"LIFTS_epoch{epoch + 1}_joint_loss{mean_loss:.6f}",
            )
        if self.best_model_dict is None:
            raise ValueError("No eligible LIFTS checkpoint was produced.")
