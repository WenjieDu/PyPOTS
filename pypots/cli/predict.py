"""
CLI command to run predictions with trained PyPOTS models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import inspect
import os

import click

from .utils import SUPPORTED_TASKS, get_model_class, load_config


@click.command(name="predict", help="Run predictions with a trained PyPOTS model")
@click.option("--model_path", required=True, type=click.Path(exists=True), help="Path to the saved .pypots model file")
@click.option("--test_set", required=True, type=click.Path(exists=True), help="Path to the test data (H5 file)")
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Path to the config file used for training (recommended, needed for correct model architecture)",
)
@click.option(
    "--task",
    type=click.Choice(SUPPORTED_TASKS),
    default=None,
    help="Task type override (read from config if not given)",
)
@click.option("--model", default=None, type=str, help="Model class name override (read from config if not given)")
@click.option(
    "--output",
    default=None,
    type=str,
    help="Path to save prediction results as an H5 file. If not given, only print a summary.",
)
@click.option("--device", default=None, type=str, help="Device override (e.g. 'cpu', 'cuda:0')")
@click.option("--file_type", default="hdf5", type=str, help="Input file type for the test set (default: hdf5)")
def predict(model_path, test_set, config, task, model, output, device, file_type):
    """Execute the predict command."""
    import numpy as np

    from ..utils.logging import logger

    # Load config if provided
    cfg = {}
    if config is not None:
        cfg = load_config(config)

    # Resolve task and model name from config or CLI args
    resolved_task = task or cfg.get("task")
    model_config = cfg.get("model", {})
    model_name = model or model_config.get("name")
    assert resolved_task is not None, "Task must be specified via --task or in the config file"
    assert model_name is not None, "Model name must be specified via --model or in the config file (model.name)"

    logger.info(f"Resolving model class '{model_name}' for task '{resolved_task}'...")
    model_class = get_model_class(resolved_task, model_name)

    # Build model kwargs from config for correct architecture
    model_kwargs = {k: v for k, v in model_config.items() if k != "name"}

    # Apply training params that are part of model constructor
    training_config = cfg.get("training", {})
    for key in ["epochs", "batch_size", "patience", "saving_path", "model_saving_strategy", "verbose"]:
        if key in training_config:
            model_kwargs[key] = training_config[key]

    # Apply device
    resolved_device = device or cfg.get("device")
    if resolved_device is not None:
        model_kwargs["device"] = resolved_device

    # Filter kwargs to only those accepted by the model's __init__
    sig = inspect.signature(model_class.__init__)
    accepted_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_keyword:
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_params}

    logger.info(f"Instantiating {model_class.__name__} for loading...")
    model_instance = model_class(**model_kwargs)

    # Restore the trained model from disk
    logger.info(f"Loading model from '{model_path}'...")
    model_instance.load(model_path)

    # Run prediction
    logger.info(f"Running prediction on '{test_set}' (file_type={file_type})...")
    results = model_instance.predict(test_set, file_type=file_type)

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
    if output is not None:
        from ..data.saving.h5 import save_dict_into_h5

        saving_dir = os.path.dirname(output) or "."
        file_name = os.path.basename(output)
        save_dict_into_h5(results, saving_dir, file_name)
        logger.info(f"Predictions saved to '{output}'")

    logger.info("Predict command completed successfully.")
