"""
CLI command for hyperparameter optimization with Optuna.
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
    get_model_init_params,
    get_optimizer_class,
)

# Mapping from config sampler names to Optuna sampler classes
_SAMPLER_MAPPING = {
    "TPE": "TPESampler",
    "Random": "RandomSampler",
    "CmaEs": "CmaEsSampler",
    "Grid": "GridSampler",
}

# Mapping from config pruner names to Optuna pruner classes
_PRUNER_MAPPING = {
    "MedianPruner": "MedianPruner",
    "PercentilePruner": "PercentilePruner",
    "HyperbandPruner": "HyperbandPruner",
    "NopPruner": "NopPruner",
}

# GAN models use G_optimizer and D_optimizer instead of a single optimizer
_GAN_OPTIMIZER_PARAMS = {"G_optimizer", "D_optimizer"}


def _suggest_param(trial, name: str, param_cfg: dict):
    """Use ``trial.suggest_*()`` to sample a hyperparameter value.

    Parameters
    ----------
    trial :
        An Optuna trial object.
    name : str
        The parameter name.
    param_cfg : dict
        The search space configuration for this parameter, containing at least ``type``.

    Returns
    -------
    value :
        The sampled value.
    """
    sp_type = param_cfg["type"]
    if sp_type == "int":
        kwargs = {"name": name, "low": param_cfg["low"], "high": param_cfg["high"]}
        if "step" in param_cfg:
            kwargs["step"] = param_cfg["step"]
        if param_cfg.get("log", False):
            kwargs["log"] = True
        return trial.suggest_int(**kwargs)
    elif sp_type == "float":
        kwargs = {"name": name, "low": param_cfg["low"], "high": param_cfg["high"]}
        if "step" in param_cfg:
            kwargs["step"] = param_cfg["step"]
        if param_cfg.get("log", False):
            kwargs["log"] = True
        return trial.suggest_float(**kwargs)
    elif sp_type == "categorical":
        return trial.suggest_categorical(name, param_cfg["choices"])
    else:
        raise ValueError(
            f"Unsupported search space type '{sp_type}' for parameter '{name}'. "
            f"Supported types: int, float, categorical"
        )


def _create_sampler(sampler_name: str, search_space: dict = None):
    """Create an Optuna sampler instance from its config name.

    Parameters
    ----------
    sampler_name : str
        Sampler name as specified in the config file (e.g. "TPE", "Random").
    search_space : dict, optional
        Required for GridSampler — maps parameter names to lists of candidate values.

    Returns
    -------
    sampler :
        An Optuna sampler instance.
    """
    import optuna

    assert sampler_name in _SAMPLER_MAPPING, (
        f"Unknown sampler '{sampler_name}'. Supported samplers: {list(_SAMPLER_MAPPING.keys())}"
    )
    cls_name = _SAMPLER_MAPPING[sampler_name]
    sampler_cls = getattr(optuna.samplers, cls_name)
    if sampler_name == "Grid":
        assert search_space is not None, "GridSampler requires the full search_space"
        # GridSampler needs a dict of {param: [values]}
        grid = {}
        for pname, pcfg in search_space.items():
            if pcfg["type"] == "categorical":
                grid[pname] = pcfg["choices"]
            else:
                raise ValueError(
                    f"GridSampler only supports categorical search spaces, "
                    f"but parameter '{pname}' has type '{pcfg['type']}'"
                )
        return sampler_cls(grid)
    return sampler_cls()


def _create_pruner(pruner_name: str):
    """Create an Optuna pruner instance from its config name.

    Parameters
    ----------
    pruner_name : str
        Pruner name as specified in the config file (e.g. "MedianPruner").

    Returns
    -------
    pruner :
        An Optuna pruner instance.
    """
    import optuna

    assert pruner_name in _PRUNER_MAPPING, (
        f"Unknown pruner '{pruner_name}'. Supported pruners: {list(_PRUNER_MAPPING.keys())}"
    )
    cls_name = _PRUNER_MAPPING[pruner_name]
    return getattr(optuna.pruners, cls_name)()


def _print_results(study):
    """Print a summary of the Optuna study results.

    Parameters
    ----------
    study :
        A completed Optuna study.
    """
    from ..utils.logging import logger

    best = study.best_trial

    logger.info("=" * 70)
    logger.info("Optuna Hyperparameter Optimization Complete")
    logger.info("=" * 70)
    logger.info(f"  Best trial number : {best.number}")
    logger.info(f"  Best value        : {best.value}")
    logger.info("  Best parameters   :")
    for k, v in best.params.items():
        logger.info(f"    {k}: {v}")

    # Summary table of all trials
    trials = study.trials
    logger.info("")
    logger.info(f"  All trials ({len(trials)} total):")
    logger.info(f"  {'Trial':>6}  {'Value':>14}  {'State':<10}  Params")
    logger.info(f"  {'-' * 6}  {'-' * 14}  {'-' * 10}  {'-' * 30}")
    for t in trials:
        value_str = f"{t.value:.6f}" if t.value is not None else "N/A"
        params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        logger.info(f"  {t.number:>6}  {value_str:>14}  {t.state.name:<10}  {params_str}")
    logger.info("=" * 70)


@click.command(name="tune", help="Run hyperparameter optimization for a PyPOTS model via Optuna")
@click.option(
    "--config", required=True, type=click.Path(exists=True), help="Path to a YAML or JSON tuning configuration file"
)
@click.option(
    "--task",
    type=click.Choice(SUPPORTED_TASKS),
    default=None,
    help="Override the task type specified in the config file",
)
@click.option("--model", default=None, type=str, help="Override the model name specified in the config file")
@click.option("--n_trials", type=int, default=None, help="Override the maximum number of tuning trials (default: 50)")
@click.option("--device", default=None, type=str, help="Override the device to use (e.g. 'cpu', 'cuda:0')")
def tune(config, task, model, n_trials, device):
    """Execute the hyperparameter optimization pipeline."""
    import optuna

    from ..utils.logging import logger

    # Suppress Optuna's verbose logging; let PyPOTS logger handle output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ------------------------------------------------------------------
    # Step 1: Load configuration from file
    # ------------------------------------------------------------------
    cfg = load_config(config)
    logger.info(f"Loaded tuning configuration from '{config}'")

    # ------------------------------------------------------------------
    # Step 2: Merge CLI overrides into the config
    # ------------------------------------------------------------------
    cfg = merge_config_with_overrides(cfg, {"task": task, "device": device})

    # Override n_trials: CLI > tuner section > default 50
    tuner_config = cfg.get("tuner", {})
    if n_trials is not None:
        resolved_n_trials = n_trials
    else:
        resolved_n_trials = tuner_config.get("n_trials", 50)

    # Override model name
    model_config = cfg.get("model", {})
    if model is not None:
        model_name = model
    else:
        model_name = model_config.get("name")
    assert model_name is not None, "Model name must be specified in the config file (model.name) or via --model"

    # Resolve task
    resolved_task = cfg.get("task")
    assert resolved_task is not None, "Task type must be specified in the config file or via --task"

    # ------------------------------------------------------------------
    # Step 3: Validate model exists and resolve class
    # ------------------------------------------------------------------
    logger.info(f"Validating model: task='{resolved_task}', model='{model_name}'")
    model_class = get_model_class(resolved_task, model_name)

    # ------------------------------------------------------------------
    # Step 4: Validate search_space param names against model __init__
    # ------------------------------------------------------------------
    search_space = cfg.get("search_space", {})
    assert search_space, "The config file must contain a non-empty 'search_space' section."

    model_params = get_model_init_params(resolved_task, model_name)
    model_param_names = set(model_params.keys())
    search_param_names = set(search_space.keys())

    # Allow 'lr' in search_space even though it's an optimizer kwarg, not a direct model param
    # lr_in_search = "lr" in search_param_names
    validate_names = search_param_names - {"lr"}
    invalid_params = validate_names - model_param_names
    if invalid_params:
        raise ValueError(
            f"Search space contains parameters not accepted by {model_name}.__init__(): "
            f"{sorted(invalid_params)}. "
            f"Valid parameters: {sorted(model_param_names)}"
        )
    logger.info(f"Search space parameters validated: {sorted(search_param_names)}")

    # ------------------------------------------------------------------
    # Step 5: Resolve data paths
    # ------------------------------------------------------------------
    data_config = cfg.get("data", {})
    train_set = data_config.get("train_set")
    val_set = data_config.get("val_set")
    assert train_set is not None, "data.train_set must be specified in the config file"
    assert val_set is not None, "data.val_set must be specified in the config file"

    resolved_device = cfg.get("device", "cpu")
    seed = cfg.get("seed", None)

    # Set random seed if provided
    if seed is not None:
        from ..utils.random import set_random_seed

        set_random_seed(seed)
        logger.info(f"Random seed set to {seed}")

    # ------------------------------------------------------------------
    # Step 6: Resolve fixed model kwargs (from model config, excluding name)
    # ------------------------------------------------------------------
    training_config = cfg.get("training", {})
    fixed_model_kwargs = {k: v for k, v in model_config.items() if k != "name"}

    # Apply training params to model kwargs
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
            fixed_model_kwargs[key] = training_config[key]

    # Set device
    if resolved_device is not None:
        fixed_model_kwargs["device"] = resolved_device

    # Resolve optimizer configuration (used when lr is NOT in the search space)
    optimizer_config = training_config.get("optimizer", None)

    # Check if the model is a GAN (has G_optimizer / D_optimizer params)
    is_gan_model = bool(_GAN_OPTIMIZER_PARAMS & model_param_names)

    # Determine the direction for optimization
    direction = tuner_config.get("direction", "minimize")
    timeout = tuner_config.get("timeout", None)

    # ------------------------------------------------------------------
    # Step 7: Create Optuna study
    # ------------------------------------------------------------------
    sampler_name = tuner_config.get("sampler", "TPE")
    sampler = _create_sampler(sampler_name, search_space)

    pruner_name = tuner_config.get("pruner", None)
    pruner = _create_pruner(pruner_name) if pruner_name else None

    study_kwargs = {
        "direction": direction,
        "sampler": sampler,
    }
    if pruner is not None:
        study_kwargs["pruner"] = pruner

    study = optuna.create_study(**study_kwargs)
    logger.info(
        f"Created Optuna study (sampler={sampler_name}, direction={direction}, "
        f"pruner={pruner_name or 'None'}, n_trials={resolved_n_trials})"
    )

    # ------------------------------------------------------------------
    # Step 8: Define objective function
    # ------------------------------------------------------------------
    def objective(trial):
        # Reset random seed before each trial to ensure reproducibility.
        # With the same seed, identical hyperparameters produce identical
        # model initializations, making trial comparisons fair.
        if seed is not None:
            from ..utils.random import set_random_seed

            set_random_seed(seed)

        # Sample hyperparameters from the search space
        trial_kwargs = {}
        sampled_lr = None
        for param_name, param_cfg in search_space.items():
            value = _suggest_param(trial, param_name, param_cfg)
            if param_name == "lr":
                sampled_lr = value
            else:
                trial_kwargs[param_name] = value

        # Merge fixed config with trial-sampled params (trial overrides fixed)
        model_kwargs = {**fixed_model_kwargs, **trial_kwargs}

        # Handle optimizer / learning rate
        if sampled_lr is not None:
            if is_gan_model:
                # GAN models: set lr on both G_optimizer and D_optimizer
                optimizer_name = "Adam"
                if optimizer_config and "name" in optimizer_config:
                    optimizer_name = optimizer_config["name"]
                opt_cls = get_optimizer_class(optimizer_name)
                model_kwargs["G_optimizer"] = opt_cls(lr=sampled_lr)
                model_kwargs["D_optimizer"] = opt_cls(lr=sampled_lr)
            else:
                # Standard models: create a single optimizer with the sampled lr
                optimizer_name = "Adam"
                if optimizer_config and "name" in optimizer_config:
                    optimizer_name = optimizer_config["name"]
                opt_cls = get_optimizer_class(optimizer_name)
                model_kwargs["optimizer"] = opt_cls(lr=sampled_lr)
        elif optimizer_config is not None:
            # lr is not being tuned but an optimizer is configured
            optimizer_name = optimizer_config.get("name", "Adam")
            opt_cls = get_optimizer_class(optimizer_name)
            opt_kwargs = {k: v for k, v in optimizer_config.items() if k != "name"}
            if is_gan_model:
                model_kwargs["G_optimizer"] = opt_cls(**opt_kwargs)
                model_kwargs["D_optimizer"] = opt_cls(**opt_kwargs)
            else:
                model_kwargs["optimizer"] = opt_cls(**opt_kwargs)

        # Pass the Optuna trial for in-training pruning
        model_kwargs["optuna_trial"] = trial

        # Filter kwargs to only those accepted by the model's __init__
        sig = inspect.signature(model_class.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_var_keyword:
            filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_params}
            skipped = set(model_kwargs.keys()) - set(filtered_kwargs.keys())
            if skipped:
                logger.debug(f"Trial {trial.number}: skipping params not accepted by {model_name}: {skipped}")
            model_kwargs = filtered_kwargs

        # Instantiate and train the model
        trial_model = model_class(**model_kwargs)
        trial_model.fit(train_set=train_set, val_set=val_set)
        return trial_model.best_loss

    # ------------------------------------------------------------------
    # Step 9: Run optimization
    # ------------------------------------------------------------------
    logger.info(f"Starting Optuna optimization ({resolved_n_trials} trials)...")
    study.optimize(objective, n_trials=resolved_n_trials, timeout=timeout)

    # ------------------------------------------------------------------
    # Step 10: Print results summary
    # ------------------------------------------------------------------
    _print_results(study)
