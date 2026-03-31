"""
CLI command to train PyPOTS models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import inspect
from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from .utils import (
    SUPPORTED_TASKS,
    load_config,
    merge_config_with_args,
    get_model_class,
    get_optimizer_class,
)
from ..utils.logging import logger


def train_command_factory(args: Namespace):
    return TrainCommand(
        config=args.config,
        task=args.task,
        model=args.model,
        train_set=args.train_set,
        val_set=args.val_set,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        saving_path=args.saving_path,
        seed=args.seed,
    )


class TrainCommand(BaseCommand):
    """CLI command for training PyPOTS models from a YAML/JSON configuration file.

    Examples
    --------
    $ pypots-cli train --config config.yaml
    $ pypots-cli train --config config.yaml --epochs 50 --device cuda:0
    $ pypots-cli train --config config.yaml --task imputation --model SAITS --seed 42
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "train",
            help="Train a PyPOTS model from a YAML/JSON configuration file",
            allow_abbrev=True,
        )

        sub_parser.add_argument(
            "--config",
            dest="config",
            type=str,
            required=True,
            help="Path to a YAML or JSON configuration file",
        )
        sub_parser.add_argument(
            "--task",
            dest="task",
            type=str,
            default=None,
            choices=SUPPORTED_TASKS,
            help="Override the task type specified in the config file",
        )
        sub_parser.add_argument(
            "--model",
            dest="model",
            type=str,
            default=None,
            help="Override the model name specified in the config file",
        )
        sub_parser.add_argument(
            "--train_set",
            dest="train_set",
            type=str,
            default=None,
            help="Override the path to training data (H5 file)",
        )
        sub_parser.add_argument(
            "--val_set",
            dest="val_set",
            type=str,
            default=None,
            help="Override the path to validation data (H5 file)",
        )
        sub_parser.add_argument(
            "--epochs",
            dest="epochs",
            type=int,
            default=None,
            help="Override the number of training epochs",
        )
        sub_parser.add_argument(
            "--batch_size",
            dest="batch_size",
            type=int,
            default=None,
            help="Override the batch size for training",
        )
        sub_parser.add_argument(
            "--device",
            dest="device",
            type=str,
            default=None,
            help="Override the device to use (e.g. 'cpu', 'cuda:0', 'mps')",
        )
        sub_parser.add_argument(
            "--saving_path",
            dest="saving_path",
            type=str,
            default=None,
            help="Override the model saving path",
        )
        sub_parser.add_argument(
            "--seed",
            dest="seed",
            type=int,
            default=None,
            help="Random seed for reproducibility",
        )

        sub_parser.set_defaults(func=train_command_factory)

    def __init__(
        self,
        config: str,
        task: str = None,
        model: str = None,
        train_set: str = None,
        val_set: str = None,
        epochs: int = None,
        batch_size: int = None,
        device: str = None,
        saving_path: str = None,
        seed: int = None,
    ):
        self._config_path = config
        self._task = task
        self._model = model
        self._train_set = train_set
        self._val_set = val_set
        self._epochs = epochs
        self._batch_size = batch_size
        self._device = device
        self._saving_path = saving_path
        self._seed = seed

    def checkup(self):
        """Validate arguments before running."""
        import os

        assert os.path.exists(self._config_path), (
            f"Configuration file not found: {self._config_path}"
        )

    def run(self):
        """Execute the training pipeline."""
        self.checkup()

        # Step 1: Load configuration from file
        config = load_config(self._config_path)
        logger.info(f"Loaded configuration from '{self._config_path}'")

        # Step 2: Merge CLI overrides into the config
        override_keys = ["task", "device", "seed"]
        config = merge_config_with_args(config, Namespace(
            task=self._task,
            device=self._device,
            seed=self._seed,
        ), override_keys)

        # Step 3: Set random seed if provided
        seed = config.get("seed", None)
        if seed is not None:
            from ..utils.random import set_random_seed
            set_random_seed(seed)
            logger.info(f"Random seed set to {seed}")

        # Step 4: Resolve task and model name
        task = config.get("task")
        assert task is not None, "Task type must be specified in the config file or via --task"

        model_config = config.get("model", {})
        model_name = self._model if self._model is not None else model_config.get("name")
        assert model_name is not None, "Model name must be specified in the config file or via --model"

        logger.info(f"Resolving model class: task='{task}', model='{model_name}'")
        model_class = get_model_class(task, model_name)

        # Step 5: Resolve optimizer if configured
        training_config = config.get("training", {})
        optimizer_config = training_config.get("optimizer", None)
        optimizer = None
        if optimizer_config is not None:
            optimizer_name = optimizer_config.get("name", "Adam")
            optimizer_cls = get_optimizer_class(optimizer_name)
            # Extract optimizer kwargs (everything except 'name')
            optimizer_kwargs = {k: v for k, v in optimizer_config.items() if k != "name"}
            optimizer = optimizer_cls(**optimizer_kwargs)
            logger.info(f"Using optimizer: {optimizer_name}")

        # Step 6: Build model constructor kwargs from config
        # Start with model architecture params (everything except 'name')
        model_kwargs = {k: v for k, v in model_config.items() if k != "name"}

        # Apply training params
        training_key_mapping = [
            "epochs", "batch_size", "patience", "saving_path", "model_saving_strategy", "verbose",
        ]
        for key in training_key_mapping:
            if key in training_config:
                model_kwargs[key] = training_config[key]

        # Apply CLI overrides for training params
        if self._epochs is not None:
            model_kwargs["epochs"] = self._epochs
        if self._batch_size is not None:
            model_kwargs["batch_size"] = self._batch_size
        if self._saving_path is not None:
            model_kwargs["saving_path"] = self._saving_path

        # Set device
        device = config.get("device", None)
        if device is not None:
            model_kwargs["device"] = device

        # Set optimizer
        if optimizer is not None:
            model_kwargs["optimizer"] = optimizer

        # Step 7: Filter kwargs to only those accepted by the model's __init__
        sig = inspect.signature(model_class.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_params}
            skipped = set(model_kwargs.keys()) - set(filtered_kwargs.keys())
            if skipped:
                logger.warning(
                    f"Skipping parameters not accepted by {model_name}: {skipped}"
                )
            model_kwargs = filtered_kwargs

        # Step 8: Instantiate the model
        logger.info(f"Instantiating model '{model_name}'...")
        model = model_class(**model_kwargs)

        # Step 8: Determine train_set and val_set
        data_config = config.get("data", {})
        train_set = self._train_set if self._train_set is not None else data_config.get("train_set")
        val_set = self._val_set if self._val_set is not None else data_config.get("val_set", None)
        assert train_set is not None, (
            "Training data path must be specified in the config file (data.train_set) or via --train_set"
        )

        # Step 9: Train the model
        logger.info(f"Starting training with train_set='{train_set}', val_set='{val_set}'")
        model.fit(train_set=train_set, val_set=val_set)

        # Step 10: Log success
        logger.info(
            f"Training complete! Model '{model_name}' for task '{task}' has been trained successfully."
        )
