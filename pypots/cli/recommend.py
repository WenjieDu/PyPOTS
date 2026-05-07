"""
CLI command to recommend model hyperparameters based on data properties.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import click


def _get_data_properties(data_path):
    """Extract data properties from an H5 or CSV file.

    Returns a dict with n_samples, n_steps, n_features, missing_rate, n_classes.
    """
    import numpy as np

    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".csv":
        import pandas as pd

        from .data import _csv_to_3d_array, _detect_columns

        df = pd.read_csv(data_path)
        sample_id_col, label_col, feature_cols = _detect_columns(df)
        X, _ = _csv_to_3d_array(df, sample_id_col, feature_cols)
        n_samples, n_steps, n_features = X.shape
        missing_rate = float(np.isnan(X).sum() / X.size)

        n_classes = None
        if label_col:
            n_classes = int(df[label_col].dropna().nunique())

        return {
            "n_samples": n_samples,
            "n_steps": n_steps,
            "n_features": n_features,
            "missing_rate": missing_rate,
            "n_classes": n_classes,
        }

    elif ext in (".h5", ".hdf5"):
        from ..data.saving.h5 import load_dict_from_h5

        loaded = load_dict_from_h5(data_path)
        X = loaded.get("X")
        if X is None:
            raise click.BadParameter(f"H5 file {data_path} does not contain 'X' key.")
        n_samples, n_steps, n_features = X.shape
        missing_rate = float(np.isnan(X).sum() / X.size)

        n_classes = None
        if "y" in loaded:
            n_classes = int(len(np.unique(loaded["y"][~np.isnan(loaded["y"])])))

        return {
            "n_samples": n_samples,
            "n_steps": n_steps,
            "n_features": n_features,
            "missing_rate": missing_rate,
            "n_classes": n_classes,
        }

    else:
        raise click.BadParameter(f"Unsupported file format '{ext}'. Supported: .csv, .h5, .hdf5")


def _recommend_hyperparams(task, model_name, n_steps, n_features, n_samples, missing_rate=0.0, n_classes=None):
    """Generate recommended hyperparameters based on data properties.

    Returns a config dict ready for YAML serialization.
    """
    # determine data scale
    if n_features <= 10:
        scale = "small"
    elif n_features <= 50:
        scale = "medium"
    else:
        scale = "large"

    # base recommendations by scale
    scale_defaults = {
        "small": {"d_model": 64, "n_layers": 2, "d_ffn": 128, "dropout": 0.1},
        "medium": {"d_model": 128, "n_layers": 2, "d_ffn": 256, "dropout": 0.1},
        "large": {"d_model": 256, "n_layers": 3, "d_ffn": 512, "dropout": 0.2},
    }
    defaults = scale_defaults[scale]

    d_model = defaults["d_model"]
    n_layers = defaults["n_layers"]
    d_ffn = defaults["d_ffn"]
    dropout = defaults["dropout"]

    # compute n_heads: largest divisor of d_model that is <= 8
    n_heads = 1
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            n_heads = h
            break

    d_k = d_model // n_heads
    d_v = d_model // n_heads

    # training parameters
    batch_size = min(32, max(8, n_samples // 8))
    # round to nearest power of 2
    batch_size = 2 ** max(3, min(5, int(batch_size).bit_length() - 1))

    if n_samples < 500:
        epochs = 200
        patience = 20
    elif n_samples < 5000:
        epochs = 100
        patience = 15
    else:
        epochs = 50
        patience = 10

    # adjust for high missing rate
    if missing_rate > 0.5:
        epochs = int(epochs * 1.5)
        patience = int(patience * 1.5)

    # build model config based on model name
    model_config = {"name": model_name}

    model_name_upper = model_name.upper()

    if model_name_upper == "SAITS":
        model_config.update(
            {
                "n_steps": n_steps,
                "n_features": n_features,
                "n_layers": n_layers,
                "d_model": d_model,
                "n_heads": n_heads,
                "d_k": d_k,
                "d_v": d_v,
                "d_ffn": d_ffn,
                "dropout": dropout,
            }
        )

    elif model_name_upper == "TIMESNET":
        top_k = min(3, n_steps // 4) if n_steps > 8 else 1
        n_kernels = 6
        model_config.update(
            {
                "n_steps": n_steps,
                "n_features": n_features,
                "n_layers": n_layers,
                "top_k": top_k,
                "d_model": d_model,
                "d_ffn": d_ffn,
                "n_kernels": n_kernels,
                "dropout": dropout,
            }
        )
        if task == "classification" and n_classes is not None:
            model_config["n_classes"] = n_classes

    elif model_name_upper == "TEFN":
        n_pred_steps = max(1, n_steps // 4)
        model_config.update(
            {
                "n_steps": n_steps,
                "n_features": n_features,
                "n_pred_steps": n_pred_steps,
                "n_pred_features": n_features,
                "n_fod": 2,
                "apply_nonstationary_norm": False,
            }
        )

    elif model_name_upper == "CRLI":
        n_clusters = n_classes if n_classes else 3
        model_config.update(
            {
                "n_steps": n_steps,
                "n_features": n_features,
                "n_clusters": n_clusters,
                "n_generator_layers": n_layers,
                "rnn_hidden_size": d_model,
                "rnn_cell_type": "GRU",
                "lambda_kmeans": 1.0,
            }
        )

    elif model_name_upper == "TIMEMIXER":
        downsampling_layers = min(3, max(1, int(n_steps).bit_length() - 3))
        model_config.update(
            {
                "n_steps": n_steps,
                "n_features": n_features,
                "n_layers": n_layers,
                "d_model": d_model,
                "d_ffn": d_ffn,
                "top_k": min(3, n_steps // 4) if n_steps > 8 else 1,
                "dropout": dropout,
                "downsampling_layers": downsampling_layers,
                "downsampling_window": 2,
            }
        )
        if task == "anomaly_detection":
            model_config["anomaly_rate"] = 0.05

    else:
        # generic fallback — include common params
        model_config.update(
            {
                "n_steps": n_steps,
                "n_features": n_features,
                "n_layers": n_layers,
                "d_model": d_model,
                "d_ffn": d_ffn,
                "dropout": dropout,
            }
        )
        if n_classes is not None:
            model_config["n_classes"] = n_classes

    # build full config
    config = {
        "task": task,
        "model": model_config,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "saving_path": f"./results/{model_name.lower()}",
            "model_saving_strategy": "best",
        },
        "data": {
            "train_set": "<path_to_train.h5>",
            "val_set": "<path_to_val.h5>",
        },
        "device": "cpu",
        "seed": 2024,
    }

    return config


# default model for each task
_DEFAULT_MODELS = {
    "imputation": "SAITS",
    "classification": "TimesNet",
    "forecasting": "TEFN",
    "clustering": "CRLI",
    "anomaly_detection": "TimeMixer",
}


@click.command(name="recommend", help="Recommend model hyperparameters based on data properties")
@click.option(
    "--task",
    required=True,
    type=click.Choice(["imputation", "classification", "forecasting", "clustering", "anomaly_detection"]),
    help="Target task type",
)
@click.option(
    "--model", "model_name", default=None, type=str, help="Model name (default: recommended model for the task)"
)
@click.option(
    "--data",
    "data_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to data file (CSV or H5) to extract properties from",
)
@click.option("--n_steps", default=None, type=int, help="Number of time steps (if not using --data)")
@click.option("--n_features", default=None, type=int, help="Number of features (if not using --data)")
@click.option("--n_samples", default=None, type=int, help="Number of samples (if not using --data)")
@click.option("--n_classes", default=None, type=int, help="Number of classes for classification/clustering")
@click.option("--output", "output_path", default=None, type=str, help="Save recommended config to YAML file")
def recommend(task, model_name, data_path, n_steps, n_features, n_samples, n_classes, output_path):
    """Recommend model hyperparameters based on data properties.

    Accepts either a data file (--data) or explicit data dimensions (--n_steps, --n_features).
    Generates a ready-to-use YAML config with recommended hyperparameters.
    """
    from ..utils.logging import logger

    # resolve model name
    if model_name is None:
        model_name = _DEFAULT_MODELS.get(task, "SAITS")
        logger.info(f"No model specified. Using default for '{task}': {model_name}")

    # get data properties
    if data_path is not None:
        logger.info(f"Extracting data properties from {data_path}")
        props = _get_data_properties(data_path)
        n_steps = props["n_steps"]
        n_features = props["n_features"]
        n_samples = props["n_samples"]
        missing_rate = props["missing_rate"]
        if props["n_classes"] is not None and n_classes is None:
            n_classes = props["n_classes"]
    else:
        if n_steps is None or n_features is None:
            raise click.UsageError("Either --data or both --n_steps and --n_features must be provided.")
        if n_samples is None:
            n_samples = 1000  # assume medium size
        missing_rate = 0.1  # assume moderate missing rate

    logger.info(
        f"Data properties: n_samples={n_samples}, n_steps={n_steps}, "
        f"n_features={n_features}, missing_rate={missing_rate:.2%}"
    )

    # generate recommendations
    config = _recommend_hyperparams(
        task=task,
        model_name=model_name,
        n_steps=n_steps,
        n_features=n_features,
        n_samples=n_samples,
        missing_rate=missing_rate,
        n_classes=n_classes,
    )

    # update data paths if data file was provided
    if data_path is not None:
        data_dir = os.path.dirname(os.path.abspath(data_path))
        data_basename = os.path.basename(data_path)
        # try to infer sibling train/val/test files
        ext = os.path.splitext(data_path)[1]
        for set_name, key in [("train", "train_set"), ("val", "val_set"), ("test", "test_set")]:
            candidate = os.path.join(data_dir, f"{set_name}{ext}")
            if os.path.exists(candidate):
                config["data"][key] = candidate
            elif set_name in data_basename.lower():
                config["data"][key] = data_path

    # save to YAML if requested
    if output_path:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save config files. Install: pip install pyyaml")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved recommended config to {output_path}")

    # print recommendation
    print(f"\n{'=' * 65}")
    print(f"Recommended Configuration: {model_name} for {task}")
    print(f"{'=' * 65}")

    print("\n  Data properties:")
    print(f"    n_samples:    {n_samples}")
    print(f"    n_steps:      {n_steps}")
    print(f"    n_features:   {n_features}")
    print(f"    missing_rate: {missing_rate:.2%}")
    if n_classes:
        print(f"    n_classes:    {n_classes}")

    print("\n  Model hyperparameters:")
    for k, v in config["model"].items():
        if k != "name":
            print(f"    {k}: {v}")

    print("\n  Training parameters:")
    for k, v in config["training"].items():
        print(f"    {k}: {v}")

    if output_path:
        print(f"\n  Config saved to: {output_path}")
        print(f"\n  To train: pypots-cli train --config {output_path}")
    else:
        print(
            f"\n  To save: pypots-cli recommend --task {task} --model {model_name} "
            f"--data <data_file> --output config.yaml"
        )

    print(f"{'=' * 65}\n")
