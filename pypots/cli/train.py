"""
CLI command to train PyPOTS models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import inspect

import click

from .utils import (
    SUPPORTED_TASKS,
    load_config,
    merge_config_with_overrides,
    get_model_class,
    get_optimizer_class,
)


@click.command(name="train", help="Train a PyPOTS model from a YAML/JSON configuration file")
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to a YAML or JSON configuration file")
@click.option(
    "--task",
    type=click.Choice(SUPPORTED_TASKS),
    default=None,
    help="Override the task type specified in the config file",
)
@click.option("--model", default=None, type=str, help="Override the model name specified in the config file")
@click.option("--train_set", default=None, type=str, help="Override the path to training data (H5 file)")
@click.option("--val_set", default=None, type=str, help="Override the path to validation data (H5 file)")
@click.option("--epochs", default=None, type=int, help="Override the number of training epochs")
@click.option("--batch_size", default=None, type=int, help="Override the batch size for training")
@click.option("--device", default=None, type=str, help="Override the device to use (e.g. 'cpu', 'cuda:0', 'mps')")
@click.option("--saving_path", default=None, type=str, help="Override the model saving path")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
def train(config, task, model, train_set, val_set, epochs, batch_size, device, saving_path, seed):
    """Execute the training pipeline."""
    from ..utils.logging import logger

    # Step 1: Load configuration from file
    cfg = load_config(config)
    logger.info(f"Loaded configuration from '{config}'")

    # Step 2: Merge CLI overrides into the config
    cfg = merge_config_with_overrides(cfg, {"task": task, "device": device, "seed": seed})

    # Step 3: Set random seed if provided
    seed_val = cfg.get("seed", None)
    if seed_val is not None:
        from ..utils.random import set_random_seed

        set_random_seed(seed_val)
        logger.info(f"Random seed set to {seed_val}")

    # Step 4: Resolve task and model name
    resolved_task = cfg.get("task")
    assert resolved_task is not None, "Task type must be specified in the config file or via --task"

    model_config = cfg.get("model", {})
    model_name = model if model is not None else model_config.get("name")
    assert model_name is not None, "Model name must be specified in the config file or via --model"

    logger.info(f"Resolving model class: task='{resolved_task}', model='{model_name}'")
    model_class = get_model_class(resolved_task, model_name)

    # Step 5: Resolve optimizer if configured
    training_config = cfg.get("training", {})
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
        "epochs",
        "batch_size",
        "patience",
        "saving_path",
        "model_saving_strategy",
        "verbose",
    ]
    for key in training_key_mapping:
        if key in training_config:
            model_kwargs[key] = training_config[key]

    # Apply CLI overrides for training params
    if epochs is not None:
        model_kwargs["epochs"] = epochs
    if batch_size is not None:
        model_kwargs["batch_size"] = batch_size
    if saving_path is not None:
        model_kwargs["saving_path"] = saving_path

    # Set device
    resolved_device = cfg.get("device", None)
    if resolved_device is not None:
        model_kwargs["device"] = resolved_device

    # Set optimizer
    if optimizer is not None:
        model_kwargs["optimizer"] = optimizer

    # Step 7: Filter kwargs to only those accepted by the model's __init__
    sig = inspect.signature(model_class.__init__)
    accepted_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_keyword:
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_params}
        skipped = set(model_kwargs.keys()) - set(filtered_kwargs.keys())
        if skipped:
            logger.warning(f"Skipping parameters not accepted by {model_name}: {skipped}")
        model_kwargs = filtered_kwargs

    # Step 8: Instantiate the model
    logger.info(f"Instantiating model '{model_name}'...")
    model_instance = model_class(**model_kwargs)

    # Step 9: Determine train_set and val_set
    data_config = cfg.get("data", {})
    resolved_train_set = train_set if train_set is not None else data_config.get("train_set")
    resolved_val_set = val_set if val_set is not None else data_config.get("val_set", None)
    assert resolved_train_set is not None, (
        "Training data path must be specified in the config file (data.train_set) or via --train_set"
    )

    # Step 10: Train the model
    logger.info(f"Starting training with train_set='{resolved_train_set}', val_set='{resolved_val_set}'")
    model_instance.fit(train_set=resolved_train_set, val_set=resolved_val_set)

    # Step 11: Log success
    logger.info(f"Training complete! Model '{model_name}' for task '{resolved_task}' has been trained successfully.")
