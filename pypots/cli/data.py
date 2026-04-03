"""
CLI command for data management operations (convert, split, describe, load, list).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import click


@click.group(name="data", help="CLI tools for data management operations (convert, split, describe, load, list)")
def data():
    """Data management operations: convert, split, describe, load, list."""
    pass


@data.command(name="convert", help="Convert data between formats (csv/npy/npz/pkl -> h5/npy/pkl)")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input file path")
@click.option("--output", "output_path", required=True, type=str, help="Output file path")
def data_convert(input_path, output_path):
    """Convert data between formats."""
    import numpy as np

    from ..utils.logging import logger

    input_ext = os.path.splitext(input_path)[1].lower()
    output_ext = os.path.splitext(output_path)[1].lower()

    logger.info(f"Converting {input_path} ({input_ext}) -> {output_path} ({output_ext})")

    # load input data
    if input_ext == ".csv":
        import pandas as pd

        df = pd.read_csv(input_path)
        data = {"X": df.values}
        logger.info(f"Loaded CSV with shape {df.shape}")
    elif input_ext == ".npy":
        data = {"X": np.load(input_path)}
        logger.info(f"Loaded NumPy array with shape {data['X'].shape}")
    elif input_ext == ".npz":
        data = dict(np.load(input_path))
        logger.info(f"Loaded NumPy archive with keys: {list(data.keys())}")
    elif input_ext == ".pkl":
        from ..data.saving.pickle import pickle_load

        data = pickle_load(input_path)
        if not isinstance(data, dict):
            data = {"X": data}
        logger.info(f"Loaded Pickle data with keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unsupported input format: {input_ext}. Supported: .csv, .npy, .npz, .pkl")

    # save output data
    if output_ext == ".h5":
        from ..data.saving.h5 import save_dict_into_h5

        save_dict_into_h5(data, output_path)
    elif output_ext == ".npy":
        np.save(output_path, data["X"])
    elif output_ext == ".pkl":
        from ..data.saving.pickle import pickle_dump

        pickle_dump(data, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_ext}. Supported: .h5, .npy, .pkl")

    logger.info(f"Successfully converted and saved to {output_path}")


@data.command(name="split", help="Split dataset into train/val/test sets")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input H5 file path")
@click.option("--output_dir", required=True, type=str, help="Output directory for saving splits")
@click.option("--train_ratio", default=0.7, type=float, help="Training set ratio (default: 0.7)")
@click.option("--val_ratio", default=0.1, type=float, help="Validation set ratio (default: 0.1)")
@click.option("--test_ratio", default=0.2, type=float, help="Test set ratio (default: 0.2)")
@click.option("--seed", default=2024, type=int, help="Random seed for reproducible splitting (default: 2024)")
def data_split(input_path, output_dir, train_ratio, val_ratio, test_ratio, seed):
    """Split dataset into train/val/test sets."""
    import numpy as np

    from ..data.saving.h5 import load_dict_from_h5, save_dict_into_h5
    from ..utils.logging import logger

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise click.BadParameter(
            f"Train/val/test ratios must sum to 1.0, but got {ratio_sum:.6f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})."
        )

    logger.info(
        f"Splitting {input_path} with ratios "
        f"train={train_ratio}, val={val_ratio}, test={test_ratio}, seed={seed}"
    )

    loaded = load_dict_from_h5(input_path)

    # determine number of samples from the "X" key
    if "X" not in loaded:
        raise ValueError("The input H5 file must contain an 'X' key to determine the number of samples.")
    n_samples = loaded["X"].shape[0]
    logger.info(f"Loaded dataset with {n_samples} samples")

    # shuffle indices
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    # compute split boundaries
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    logger.info(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # create split dicts by indexing all arrays
    train_data, val_data, test_data = {}, {}, {}
    for key, value in loaded.items():
        if isinstance(value, np.ndarray) and value.shape[0] == n_samples:
            train_data[key] = value[train_indices]
            val_data[key] = value[val_indices]
            test_data[key] = value[test_indices]
        else:
            # for non-array or non-sample-aligned data, keep as-is in all splits
            train_data[key] = value
            val_data[key] = value
            test_data[key] = value

    # save splits
    os.makedirs(output_dir, exist_ok=True)
    save_dict_into_h5(train_data, os.path.join(output_dir, "train.h5"))
    save_dict_into_h5(val_data, os.path.join(output_dir, "val.h5"))
    save_dict_into_h5(test_data, os.path.join(output_dir, "test.h5"))

    logger.info(f"Successfully saved splits to {output_dir}/{{train,val,test}}.h5")


@data.command(name="describe", help="Inspect and describe dataset statistics")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input H5 file path")
def data_describe(input_path):
    """Inspect and describe dataset statistics."""
    import numpy as np

    from ..data.saving.h5 import load_dict_from_h5
    from ..utils.logging import logger

    logger.info(f"Describing dataset: {input_path}")

    loaded = load_dict_from_h5(input_path)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {input_path}")
    print(f"{'=' * 60}")
    print(f"Number of keys: {len(loaded)}")
    print(f"{'-' * 60}")

    n_samples = None
    n_features = None
    seq_length = None

    for key, value in loaded.items():
        if isinstance(value, np.ndarray):
            print(f"\n  Key: '{key}'")
            print(f"    dtype: {value.dtype}")
            print(f"    shape: {value.shape}")

            if np.issubdtype(value.dtype, np.number):
                total_elements = value.size
                nan_count = np.isnan(value).sum() if np.issubdtype(value.dtype, np.floating) else 0
                missing_rate = nan_count / total_elements if total_elements > 0 else 0.0

                print(f"    min: {np.nanmin(value):.6g}")
                print(f"    max: {np.nanmax(value):.6g}")
                print(f"    mean: {np.nanmean(value):.6g}")
                print(f"    std: {np.nanstd(value):.6g}")
                print(f"    missing rate: {missing_rate:.4%} ({nan_count}/{total_elements})")

            # extract dataset dimensions from "X"
            if key == "X":
                n_samples = value.shape[0]
                if value.ndim >= 3:
                    seq_length = value.shape[1]
                    n_features = value.shape[2]
                elif value.ndim == 2:
                    n_features = value.shape[1]
        else:
            print(f"\n  Key: '{key}'")
            print(f"    type: {type(value).__name__}")
            print(f"    value: {value}")

    print(f"\n{'-' * 60}")
    print("Summary:")
    if n_samples is not None:
        print(f"  Total samples: {n_samples}")
    if seq_length is not None:
        print(f"  Sequence length: {seq_length}")
    if n_features is not None:
        print(f"  Number of features: {n_features}")
    print(f"{'=' * 60}\n")


@data.command(name="list", help="List available benchmark datasets")
@click.option("--task", default=None, type=str, help="Task type filter for listing datasets")
def data_list(task):
    """List available benchmark datasets."""
    try:
        import tsdb
    except ImportError:
        raise ImportError(
            "The 'tsdb' package is required for listing benchmark datasets. "
            "Install it with: pip install tsdb"
        )

    available = tsdb.list()
    print(f"\n{'=' * 60}")
    print("Available Benchmark Datasets")
    print(f"{'=' * 60}")
    print(f"Total: {len(available)} datasets\n")
    for i, name in enumerate(available, 1):
        print(f"  {i:3d}. {name}")
    print(f"\n{'=' * 60}")
    print("Use 'pypots-cli data load --dataset <name> --output_dir <dir>' to load a dataset.")
    print(f"{'=' * 60}\n")


@data.command(name="load", help="Load a benchmark dataset and save as train/val/test H5 files")
@click.option("--dataset", required=True, type=str, help="Benchmark dataset name to load (e.g., physionet_2012)")
@click.option("--output_dir", required=True, type=str, help="Output directory for saving dataset splits")
@click.option("--subset", default=None, type=str, help="Dataset subset to load (e.g., set-a for physionet_2012)")
@click.option("--rate", default=0.1, type=float, help="Artificially missing rate for benchmark dataset (default: 0.1)")
@click.option("--n_steps", default=None, type=int, help="Number of time steps for benchmark dataset")
@click.option("--pattern", default="point", type=click.Choice(["point", "subseq", "block"]),
              help="Missing pattern for benchmark dataset (default: point)")
def data_load(dataset, output_dir, subset, rate, n_steps, pattern):
    """Load a benchmark dataset via benchpots and save as train/val/test H5 files."""
    import numpy as np

    from ..utils.logging import logger

    try:
        import benchpots.datasets as bpd
    except ImportError:
        raise ImportError(
            "The 'benchpots' package is required for loading benchmark datasets. "
            "Install it with: pip install benchpots"
        )

    dataset_name = dataset
    logger.info(f"Loading benchmark dataset: {dataset_name}")

    # map dataset names to benchpots preprocessing functions
    preprocess_map = {
        "physionet_2012": bpd.preprocess_physionet2012,
        "physionet_2019": bpd.preprocess_physionet2019,
        "beijing_air_quality": bpd.preprocess_beijing_air_quality,
        "electricity_load_diagrams": bpd.preprocess_electricity_load_diagrams,
        "italy_air_quality": bpd.preprocess_italy_air_quality,
        "pems_traffic": bpd.preprocess_pems_traffic,
        "solar_alabama": bpd.preprocess_solar_alabama,
    }

    # check for ETT datasets (ett_h1, ett_h2, ett_m1, ett_m2)
    if dataset_name.startswith("ett_"):
        preprocess_func = bpd.preprocess_ett
    elif dataset_name in preprocess_map:
        preprocess_func = preprocess_map[dataset_name]
    else:
        # try UCR/UEA datasets or other benchpots datasets
        try:
            preprocess_func = bpd.preprocess_ucr_uea_datasets
        except AttributeError:
            raise ValueError(
                f"Dataset '{dataset_name}' is not directly supported. "
                f"Supported datasets: {', '.join(sorted(preprocess_map.keys()))} and UCR/UEA datasets. "
                f"Use 'pypots-cli data list' to see all available datasets."
            )

    # build preprocessing kwargs
    kwargs = {"rate": rate, "pattern": pattern}

    # add n_steps if provided and supported
    if n_steps is not None:
        kwargs["n_steps"] = n_steps

    # handle dataset-specific parameters
    import inspect

    sig = inspect.signature(preprocess_func)
    func_params = set(sig.parameters.keys())

    if "subset" in func_params and subset is not None:
        kwargs["subset"] = subset
    elif "subset" in func_params and subset is None:
        # default to 'all' for datasets that accept subset
        kwargs["subset"] = "all"

    if dataset_name.startswith("ett_"):
        kwargs["subset"] = dataset_name.replace("_", "-")  # ett_h1 -> ett-h1

    # for UCR/UEA and NL benchmark datasets, pass dataset_name
    if "dataset_name" in func_params:
        kwargs["dataset_name"] = dataset_name

    # remove kwargs not accepted by the function
    kwargs = {k: v for k, v in kwargs.items() if k in func_params}

    logger.info(f"Preprocessing with params: {kwargs}")
    result = preprocess_func(**kwargs)

    # save as H5 files
    from ..data.saving.h5 import save_dict_into_h5

    os.makedirs(output_dir, exist_ok=True)

    # organize output by train/val/test splits
    train_data, val_data, test_data = {}, {}, {}
    metadata = {}

    for key, value in result.items():
        if isinstance(value, np.ndarray):
            if key.startswith("train_"):
                train_data[key.replace("train_", "")] = value
            elif key.startswith("val_"):
                val_data[key.replace("val_", "")] = value
            elif key.startswith("test_"):
                test_data[key.replace("test_", "")] = value
        elif key in ("n_steps", "n_features", "n_classes", "n_clusters"):
            metadata[key] = value

    # save splits
    if train_data:
        save_dict_into_h5(train_data, os.path.join(output_dir, "train.h5"))
        logger.info(f"Saved training set with keys: {list(train_data.keys())}")
    if val_data:
        save_dict_into_h5(val_data, os.path.join(output_dir, "val.h5"))
        logger.info(f"Saved validation set with keys: {list(val_data.keys())}")
    if test_data:
        save_dict_into_h5(test_data, os.path.join(output_dir, "test.h5"))
        logger.info(f"Saved test set with keys: {list(test_data.keys())}")

    # print dataset summary
    print(f"\n{'=' * 60}")
    print(f"Benchmark Dataset: {dataset_name}")
    print(f"{'=' * 60}")
    if metadata:
        for k, v in metadata.items():
            print(f"  {k}: {v}")
    if train_data and "X" in train_data:
        print(f"  Train samples: {train_data['X'].shape[0]}")
    if val_data and "X" in val_data:
        print(f"  Val samples: {val_data['X'].shape[0]}")
    if test_data and "X" in test_data:
        print(f"  Test samples: {test_data['X'].shape[0]}")
    print(f"\n  Saved to: {output_dir}/")
    print(f"    train.h5  val.h5  test.h5")
    print(f"{'=' * 60}\n")
