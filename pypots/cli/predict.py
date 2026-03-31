"""
CLI command to run predictions with trained PyPOTS models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import inspect
import os
from argparse import ArgumentParser, Namespace

import numpy as np

from .base import BaseCommand
from .utils import SUPPORTED_TASKS, get_model_class, load_config
from ..utils.logging import logger


def predict_command_factory(args: Namespace):
    return PredictCommand(
        model_path=args.model_path,
        test_set=args.test_set,
        task=args.task,
        model=args.model,
        config=args.config,
        output=args.output,
        device=args.device,
        file_type=args.file_type,
    )


class PredictCommand(BaseCommand):
    """Run predictions using a saved PyPOTS model.

    The model architecture must be known to load weights correctly. This is achieved by providing
    the same config file used during training (recommended) or by relying on the model having
    no required architecture parameters (e.g. naive models like Mean/Median).

    Examples
    --------
    $ pypots-cli predict --config train_config.yaml --model_path ./model.pypots --test_set ./test.h5
    $ pypots-cli predict --config train_config.yaml --model_path ./model.pypots --test_set ./test.h5 \
          --output ./predictions.h5 --device cuda:0
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "predict",
            help="Run predictions with a trained PyPOTS model",
            allow_abbrev=True,
        )

        sub_parser.add_argument(
            "--model_path",
            dest="model_path",
            type=str,
            required=True,
            help="Path to the saved .pypots model file",
        )
        sub_parser.add_argument(
            "--test_set",
            dest="test_set",
            type=str,
            required=True,
            help="Path to the test data (H5 file)",
        )
        sub_parser.add_argument(
            "--config",
            dest="config",
            type=str,
            default=None,
            help="Path to the config file used for training (recommended, needed for correct model architecture)",
        )
        sub_parser.add_argument(
            "--task",
            dest="task",
            type=str,
            default=None,
            choices=SUPPORTED_TASKS,
            help="Task type override (read from config if not given)",
        )
        sub_parser.add_argument(
            "--model",
            dest="model",
            type=str,
            default=None,
            help="Model class name override (read from config if not given)",
        )
        sub_parser.add_argument(
            "--output",
            dest="output",
            type=str,
            default=None,
            help="Path to save prediction results as an H5 file. If not given, only print a summary.",
        )
        sub_parser.add_argument(
            "--device",
            dest="device",
            type=str,
            default=None,
            help="Device override (e.g. 'cpu', 'cuda:0')",
        )
        sub_parser.add_argument(
            "--file_type",
            dest="file_type",
            type=str,
            default="hdf5",
            help="Input file type for the test set (default: hdf5)",
        )

        sub_parser.set_defaults(func=predict_command_factory)

    def __init__(
        self,
        model_path: str,
        test_set: str,
        task: str = None,
        model: str = None,
        config: str = None,
        output: str = None,
        device: str = None,
        file_type: str = "hdf5",
    ):
        self._model_path = model_path
        self._test_set = test_set
        self._task = task
        self._model = model
        self._config = config
        self._output = output
        self._device = device
        self._file_type = file_type

    def checkup(self):
        """Validate arguments before running."""
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        if not os.path.exists(self._test_set):
            raise FileNotFoundError(f"Test set file not found: {self._test_set}")

    def run(self):
        """Execute the predict command."""
        self.checkup()

        # Load config if provided
        config = {}
        if self._config is not None:
            config = load_config(self._config)

        # Resolve task and model name from config or CLI args
        task = self._task or config.get("task")
        model_config = config.get("model", {})
        model_name = self._model or model_config.get("name")
        assert task is not None, (
            "Task must be specified via --task or in the config file"
        )
        assert model_name is not None, (
            "Model name must be specified via --model or in the config file (model.name)"
        )

        logger.info(f"Resolving model class '{model_name}' for task '{task}'...")
        model_class = get_model_class(task, model_name)

        # Build model kwargs from config for correct architecture
        model_kwargs = {k: v for k, v in model_config.items() if k != "name"}

        # Apply training params that are part of model constructor
        training_config = config.get("training", {})
        for key in ["epochs", "batch_size", "patience", "saving_path", "model_saving_strategy", "verbose"]:
            if key in training_config:
                model_kwargs[key] = training_config[key]

        # Apply device
        device = self._device or config.get("device")
        if device is not None:
            model_kwargs["device"] = device

        # Filter kwargs to only those accepted by the model's __init__
        sig = inspect.signature(model_class.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            model_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_params}

        logger.info(f"Instantiating {model_class.__name__} for loading...")
        model = model_class(**model_kwargs)

        # Restore the trained model from disk
        logger.info(f"Loading model from '{self._model_path}'...")
        model.load(self._model_path)

        # Run prediction
        logger.info(f"Running prediction on '{self._test_set}' (file_type={self._file_type})...")
        results = model.predict(self._test_set, file_type=self._file_type)

        # Print a summary of results
        logger.info("Prediction finished. Results summary:")
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                logger.info(f"  {key}: ndarray, shape={value.shape}, dtype={value.dtype}")
            elif hasattr(value, "shape"):
                logger.info(f"  {key}: {type(value).__name__}, shape={value.shape}")
            else:
                logger.info(f"  {key}: {type(value).__name__}, value={value}")

        # Optionally save results to H5
        if self._output is not None:
            from ..data.saving.h5 import save_dict_into_h5

            saving_dir = os.path.dirname(self._output) or "."
            file_name = os.path.basename(self._output)
            save_dict_into_h5(results, saving_dir, file_name)
            logger.info(f"Predictions saved to '{self._output}'")

        logger.info("Predict command completed successfully.")
