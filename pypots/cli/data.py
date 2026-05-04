"""
CLI command for data management operations (profile, prepare, reconstruct, convert, split, describe, load, list).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import click


# Reserved column names in the ai4ts data protocol
_RESERVED_COLUMNS = {"SAMPLE_ID", "TIMESTAMP", "SAMPLE_LABEL", "STEP_LABEL"}


def _detect_columns(df):
    """Auto-detect column roles from a DataFrame following the ai4ts data protocol.

    Returns
    -------
    sample_id_col : str or None
        The SAMPLE_ID column name, or None if not found.
    label_col : str or None
        The classification target column name (containing 'CLAF_TARGET'), or None.
    feature_cols : list of str
        The numeric feature column names.
    """
    import numpy as np

    columns = list(df.columns)

    sample_id_col = "SAMPLE_ID" if "SAMPLE_ID" in columns else None

    label_col = None
    for col in columns:
        if "CLAF_TARGET" in col:
            label_col = col
            break

    reserved = _RESERVED_COLUMNS | ({label_col} if label_col else set())
    feature_cols = [
        c for c in columns
        if c not in reserved and np.issubdtype(df[c].dtype, np.number)
    ]

    return sample_id_col, label_col, feature_cols


def _csv_to_3d_array(df, sample_id_col, feature_cols):
    """Convert a DataFrame to a 3D numpy array [n_samples, n_steps, n_features].

    Groups rows by sample_id_col. If sample_id_col is None, treats all rows as one sample.
    Pads to the maximum timestep length across samples.

    Returns
    -------
    X : np.ndarray, shape [n_samples, max_steps, n_features]
    sample_ids : list
        Unique sample IDs in order.
    """
    import numpy as np

    if sample_id_col is not None and sample_id_col in df.columns:
        groups = df.groupby(sample_id_col, sort=True)
        sample_ids = sorted(df[sample_id_col].unique())
    else:
        groups = [(0, df)]
        sample_ids = [0]

    n_features = len(feature_cols)
    samples = []

    for sid in sample_ids:
        if sample_id_col is not None:
            group = groups.get_group(sid)
        else:
            group = df
        arr = group[feature_cols].values.astype(np.float64)
        samples.append(arr)

    # pad to max length
    max_steps = max(s.shape[0] for s in samples)
    n_samples = len(samples)

    X = np.full((n_samples, max_steps, n_features), np.nan, dtype=np.float64)
    for i, s in enumerate(samples):
        X[i, :s.shape[0], :] = s

    return X, sample_ids


def _extract_labels(df, sample_id_col, label_col):
    """Extract per-sample classification labels from a DataFrame.

    Returns y as a 1D numpy array [n_samples] with one label per unique sample.
    """
    import numpy as np

    if sample_id_col is not None and sample_id_col in df.columns:
        # take the first label value per sample (should be constant within a sample)
        labels = df.groupby(sample_id_col, sort=True)[label_col].first()
        y = labels.values
    else:
        # single sample, take the first label
        y = np.array([df[label_col].iloc[0]])

    # encode string labels as integers if needed
    if y.dtype == object or y.dtype.kind in ("U", "S"):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    return y.astype(np.float64)


def _prepare_single_csv(input_path, output_path, task, set_type, missing_rate, seed, logger):
    """Core logic to prepare a single CSV file into PyPOTS H5 format."""
    import numpy as np
    import pandas as pd

    from ..data.saving.h5 import save_dict_into_h5

    df = pd.read_csv(input_path)
    sample_id_col, label_col, feature_cols = _detect_columns(df)

    if not feature_cols:
        raise click.BadParameter(
            f"No numeric feature columns found in {input_path}. "
            f"Columns found: {list(df.columns)}"
        )

    logger.info(
        f"Detected columns — sample_id: {sample_id_col}, label: {label_col}, "
        f"features ({len(feature_cols)}): {feature_cols}"
    )

    # build 3D array
    X_ori, sample_ids = _csv_to_3d_array(df, sample_id_col, feature_cols)
    n_samples, n_steps, n_features = X_ori.shape

    # compute natural missing rate
    natural_missing_rate = np.isnan(X_ori).sum() / X_ori.size

    # build output dict
    data_dict = {}
    data_dict["X_ori"] = X_ori.copy()

    if set_type == "train":
        # for training: X = X_ori (raw data with natural NaN only)
        # models like SAITS add artificial missing dynamically during training
        data_dict["X"] = X_ori.copy()
    else:
        # for val/test: inject artificial missing into X
        rng = np.random.RandomState(seed)
        X_with_missing = X_ori.copy()
        if missing_rate > 0:
            # only mask positions that are NOT already NaN
            observed_mask = ~np.isnan(X_with_missing)
            # generate random mask: True = will be artificially masked
            random_mask = rng.rand(*X_with_missing.shape) < missing_rate
            artificial_mask = observed_mask & random_mask
            X_with_missing[artificial_mask] = np.nan
        data_dict["X"] = X_with_missing

    # extract classification labels if present
    if label_col is not None:
        y = _extract_labels(df, sample_id_col, label_col)
        data_dict["y"] = y
        n_classes = len(np.unique(y[~np.isnan(y)])) if np.issubdtype(y.dtype, np.floating) else len(np.unique(y))
    else:
        n_classes = None

    # save to H5
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_dict_into_h5(data_dict, output_path)

    # compute actual missing rate in X
    actual_missing_rate = np.isnan(data_dict["X"]).sum() / data_dict["X"].size

    summary = {
        "n_samples": n_samples,
        "n_steps": n_steps,
        "n_features": n_features,
        "natural_missing_rate": float(natural_missing_rate),
        "actual_missing_rate": float(actual_missing_rate),
        "has_labels": label_col is not None,
        "n_classes": n_classes,
        "set_type": set_type,
        "keys_saved": list(data_dict.keys()),
    }

    return summary


@click.group(name="data", help="CLI tools for data management operations (profile, prepare, reconstruct, convert, split, describe, load, list)")
def data():
    """Data management operations: profile, prepare, reconstruct, convert, split, describe, load, list."""
    pass


@data.command(name="profile", help="Analyze a CSV dataset and output a DataProfile JSON for agent consumption")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Input CSV file path")
@click.option("--task", default=None,
              type=click.Choice(["imputation", "classification", "forecasting", "clustering", "anomaly_detection"]),
              help="Task type hint (auto-detected if omitted)")
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Output raw JSON (default: human-readable summary)")
def data_profile(input_path, task, json_output):
    """Analyze a CSV dataset and generate a DataProfile.

    The DataProfile is a lightweight JSON structure describing the dataset's
    statistics, column schema, timestamp regularity, and sample lengths.
    Agents use this to make informed decisions about preprocessing strategy.
    """
    import json as json_mod

    from ..utils.logging import logger

    try:
        from ai4ts.data_protocols.profile import analyze_dataset
    except ImportError:
        raise click.ClickException(
            "The 'ai4ts' package is required for data profiling. "
            "Install it with: pip install ai4ts"
        )

    logger.info(f"Profiling dataset: {input_path}")
    profile = analyze_dataset(input_path, task_type=task)
    d = profile.to_dict()

    if json_output:
        print(json_mod.dumps(d, indent=2))
    else:
        stats = d["dataset_stats"]
        schema = d["schema_mapping"]
        ts_info = d["timestamp_info"]

        print(f"\n{'=' * 65}")
        print(f"Data Profile: {input_path}")
        print(f"{'=' * 65}")
        print(f"  Samples:          {stats['n_samples']}")
        print(f"  Features:         {stats['n_features']}")
        print(f"  Total rows:       {stats['total_rows']}")
        print(f"  Missing rate:     {stats['missing_rate']:.2%}")
        print(f"  Sample lengths:   min={stats['min_sample_length']}, "
              f"max={stats['max_sample_length']}, "
              f"avg={stats['avg_sample_length']:.1f}")
        print(f"  Variable length:  {stats['has_variable_length']}")
        print(f"\n  Schema:")
        print(f"    SAMPLE_ID:      {schema['sample_id_col'] or '(not present)'}")
        print(f"    TIMESTAMP:      {schema['timestamp_col'] or '(not present)'}")
        print(f"    Features:       {schema['feature_cols']}")
        print(f"    Label column:   {schema['label_col'] or '(none)'}")
        if schema['n_classes']:
            print(f"    Classes:        {schema['n_classes']}")
        print(f"\n  Task type:        {d['task_type']}")

        if ts_info.get("warning"):
            print(f"\n  ⚠ WARNING: {ts_info['warning']}")

        # Suggest appropriate strategy
        if stats['has_variable_length'] or stats['max_sample_length'] > 200:
            if d['task_type'] == 'classification':
                print(f"\n  → Strategy: pad_only (pad to max_len={stats['max_sample_length']})")
            else:
                print(f"\n  → Strategy: sliding_window (non-overlapping, window_size≤200)")
        else:
            print(f"\n  → Strategy: direct (uniform length ≤ 200)")

        print(f"\n{'=' * 65}")
        print(f"Next: pypots-cli data prepare --input {input_path} --output data.h5 --task {d['task_type'] or 'imputation'}")
        print(f"{'=' * 65}\n")


@data.command(name="prepare", help="Convert CSV data to PyPOTS H5 format with proper data structure for model training")
@click.option("--input", "input_path", default=None, type=click.Path(exists=True),
              help="Single input CSV file path (use with --output and --set_type)")
@click.option("--output", "output_path", default=None, type=str,
              help="Single output H5 file path (use with --input)")
@click.option("--train", "train_path", default=None, type=click.Path(exists=True),
              help="Training CSV file path (batch mode)")
@click.option("--val", "val_path", default=None, type=click.Path(exists=True),
              help="Validation CSV file path (batch mode)")
@click.option("--test", "test_path", default=None, type=click.Path(exists=True),
              help="Test CSV file path (batch mode)")
@click.option("--output_dir", default=None, type=str,
              help="Output directory for batch mode (saves train.h5, val.h5, test.h5)")
@click.option("--task", required=True,
              type=click.Choice(["imputation", "classification", "forecasting", "clustering", "anomaly_detection"]),
              help="Target task type (determines which keys to generate in H5)")
@click.option("--set_type", default=None, type=click.Choice(["train", "val", "test"]),
              help="Dataset split type (single-file mode). If not specified, auto-detected from filename.")
@click.option("--missing_rate", default=0.1, type=float,
              help="Artificial missing rate for val/test sets (default: 0.1)")
@click.option("--window_size", default=None, type=int,
              help="Fixed window size for sliding window (auto-determined if omitted)")
@click.option("--seed", default=2024, type=int, help="Random seed for reproducibility (default: 2024)")
def data_prepare(input_path, output_path, train_path, val_path, test_path,
                 output_dir, task, set_type, missing_rate, window_size, seed):
    """Prepare CSV data for PyPOTS model training.

    Converts CSV files following the ai4ts data protocol (with SAMPLE_ID, features,
    and optional CLAF_TARGET columns) into PyPOTS-compatible HDF5 format with proper
    3D arrays (n_samples, n_steps, n_features) and all required keys (X, X_ori, y).

    Handles variable-length samples via:
    - Classification tasks: pad to max length with NaN (no sliding window)
    - Other tasks: non-overlapping sliding window if samples are long or variable-length

    A window registry (JSON) is saved alongside each H5 file for reconstruction.

    Supports two modes:
      - Single-file: --input + --output + --set_type
      - Batch: --train/--val/--test + --output_dir (recommended)
    """
    from ..utils.logging import logger

    batch_mode = any([train_path, val_path, test_path])
    single_mode = input_path is not None

    if not batch_mode and not single_mode:
        raise click.UsageError(
            "Either provide --input (single-file mode) or --train/--val/--test (batch mode)."
        )

    if batch_mode and single_mode:
        raise click.UsageError(
            "Cannot mix single-file mode (--input) with batch mode (--train/--val/--test)."
        )

    if batch_mode and output_dir is None:
        raise click.UsageError("--output_dir is required for batch mode.")

    if single_mode and output_path is None:
        raise click.UsageError("--output is required for single-file mode.")

    # Try to use the ai4ts pipeline if available (supports windowing + registry)
    use_pipeline = _check_ai4ts_pipeline()

    all_summaries = []

    if batch_mode:
        os.makedirs(output_dir, exist_ok=True)
        files = [
            ("train", train_path),
            ("val", val_path),
            ("test", test_path),
        ]
        for st, fpath in files:
            if fpath is None:
                continue
            out_h5 = os.path.join(output_dir, f"{st}.h5")
            logger.info(f"Preparing {st} set: {fpath} -> {out_h5}")
            if use_pipeline:
                summary = _prepare_with_pipeline(
                    fpath, out_h5, task, st, missing_rate, window_size, seed, logger
                )
            else:
                summary = _prepare_single_csv(
                    fpath, out_h5, task, st, missing_rate, seed, logger
                )
            all_summaries.append(summary)
            logger.info(f"  {st}: {summary['n_samples']} samples → "
                        f"{summary.get('n_windows', summary['n_samples'])} windows, "
                        f"{summary['n_steps']} steps, {summary['n_features']} features")

    else:
        # single-file mode — auto-detect set_type from filename if not specified
        if set_type is None:
            basename = os.path.basename(input_path).lower()
            if "train" in basename:
                set_type = "train"
            elif "val" in basename:
                set_type = "val"
            elif "test" in basename:
                set_type = "test"
            else:
                set_type = "train"
                logger.warning(
                    f"Could not auto-detect set_type from filename '{basename}'. "
                    f"Defaulting to 'train'. Use --set_type to specify explicitly."
                )

        logger.info(f"Preparing {set_type} set: {input_path} -> {output_path}")
        if use_pipeline:
            summary = _prepare_with_pipeline(
                input_path, output_path, task, set_type, missing_rate, window_size, seed, logger
            )
        else:
            summary = _prepare_single_csv(
                input_path, output_path, task, set_type, missing_rate, seed, logger
            )
        all_summaries.append(summary)

    # print summary
    print(f"\n{'=' * 65}")
    print(f"Data Preparation Complete — Task: {task}")
    print(f"{'=' * 65}")
    for s in all_summaries:
        print(f"\n  [{s['set_type'].upper()}]")
        print(f"    Samples:   {s['n_samples']}")
        n_windows = s.get("n_windows", s["n_samples"])
        if n_windows != s["n_samples"]:
            print(f"    Windows:   {n_windows} (from sliding window)")
        print(f"    Steps:     {s['n_steps']}")
        print(f"    Features:  {s['n_features']}")
        print(f"    Strategy:  {s.get('strategy', 'direct')}")
        print(f"    Natural missing rate:  {s['natural_missing_rate']:.2%}")
        print(f"    Actual missing rate:   {s['actual_missing_rate']:.2%}")
        if s['has_labels']:
            print(f"    Labels:    yes ({s['n_classes']} classes)")
        print(f"    H5 keys:   {s['keys_saved']}")
        if s.get("registry_path"):
            print(f"    Registry:  {s['registry_path']}")

    if batch_mode:
        print(f"\n  Output directory: {output_dir}/")

    # print recommended next steps
    print(f"\n{'=' * 65}")
    print("Next steps:")
    if batch_mode:
        desc_target = os.path.join(output_dir, "train.h5")
    else:
        desc_target = output_path
    print(f"  1. Inspect: pypots-cli data describe --input {desc_target}")
    print(f"  2. Recommend: pypots-cli recommend --task {task} --data {desc_target}")
    print(f"  3. Train:   pypots-cli train --config <config.yaml>")
    print(f"{'=' * 65}\n")


def _check_ai4ts_pipeline() -> bool:
    """Check if the ai4ts pipeline module is available."""
    try:
        from ai4ts.data_protocols.pipeline import prepare_for_pypots  # noqa: F401
        return True
    except ImportError:
        return False


def _prepare_with_pipeline(input_path, output_path, task, set_type, missing_rate,
                           window_size, seed, logger):
    """Prepare using ai4ts pipeline with windowing and registry support."""
    import numpy as np

    from ai4ts.data_protocols.pipeline import prepare_for_pypots

    from ..data.saving.h5 import save_dict_into_h5

    result = prepare_for_pypots(
        csv_path=input_path,
        task=task,
        window_size=window_size,
        missing_rate=missing_rate,
        set_type=set_type,
        seed=seed,
    )

    X = result["X"]
    X_ori = result["X_ori"]
    registry = result["registry"]
    profile = result["profile"]

    n_windows, n_steps, n_features = X.shape
    n_samples = profile.dataset_stats.n_samples

    # Build H5 data dict
    data_dict = {"X": X, "X_ori": X_ori}
    if "y" in result:
        data_dict["y"] = result["y"]

    # Save H5
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_dict_into_h5(data_dict, output_path)

    # Save registry alongside the H5 file
    registry_path = os.path.splitext(output_path)[0] + "_registry.json"
    registry.to_json(registry_path)
    logger.info(f"  Registry saved: {registry_path}")

    natural_missing_rate = np.isnan(X_ori).sum() / X_ori.size
    actual_missing_rate = np.isnan(X).sum() / X.size

    return {
        "n_samples": n_samples,
        "n_windows": n_windows,
        "n_steps": n_steps,
        "n_features": n_features,
        "strategy": registry.strategy,
        "natural_missing_rate": float(natural_missing_rate),
        "actual_missing_rate": float(actual_missing_rate),
        "has_labels": "y" in result,
        "n_classes": profile.schema_mapping.n_classes,
        "set_type": set_type,
        "keys_saved": list(data_dict.keys()),
        "registry_path": registry_path,
    }


@data.command(name="reconstruct", help="Reconstruct original-shape data from model predictions using a window registry")
@click.option("--predictions", required=True, type=click.Path(exists=True),
              help="Path to predictions H5 file (must have 'X' key with 3D array)")
@click.option("--registry", "registry_path", required=True, type=click.Path(exists=True),
              help="Path to window registry JSON file (generated by 'data prepare')")
@click.option("--output", "output_path", required=True, type=str,
              help="Output CSV file path for reconstructed data")
@click.option("--key", default="X", type=str,
              help="Key in the H5 file containing the predictions (default: X)")
def data_reconstruct(predictions, registry_path, output_path, key):
    """Reconstruct original-shape time series from model predictions.

    After running model inference (e.g., imputation) on windowed data,
    this command reverses the windowing transformation:
    1. Loads the predictions (3D tensor) from an H5 file
    2. Reads the window registry (JSON) that was saved during 'data prepare'
    3. Strips padding from each window
    4. Reassembles windows belonging to the same original sample
    5. Outputs a CSV file with SAMPLE_ID and reconstructed features
    """
    import numpy as np
    import pandas as pd

    from ..utils.logging import logger

    try:
        from ai4ts.data_protocols.registry import WindowRegistry
        from ai4ts.data_protocols.pipeline import reconstruct_from_predictions
    except ImportError:
        raise click.ClickException(
            "The 'ai4ts' package is required for data reconstruction. "
            "Install it with: pip install ai4ts"
        )

    from ..data.saving.h5 import load_dict_from_h5

    logger.info(f"Loading predictions from: {predictions}")
    loaded = load_dict_from_h5(predictions)
    if key not in loaded:
        raise click.ClickException(
            f"Key '{key}' not found in {predictions}. Available keys: {list(loaded.keys())}"
        )
    pred_array = loaded[key]
    logger.info(f"Predictions shape: {pred_array.shape}")

    logger.info(f"Loading registry from: {registry_path}")
    registry = WindowRegistry.from_json(registry_path)
    logger.info(f"Registry: {registry.n_windows} windows, strategy={registry.strategy}")

    # Reconstruct
    reconstructed = reconstruct_from_predictions(pred_array, registry)
    logger.info(f"Reconstructed {len(reconstructed)} samples")

    # Convert to CSV with SAMPLE_ID
    frames = []
    for sample_id in sorted(reconstructed.keys()):
        arr = reconstructed[sample_id]
        n_features = arr.shape[1]
        feature_names = [f"feat_{i}" for i in range(n_features)]
        sample_df = pd.DataFrame(arr, columns=feature_names)
        sample_df.insert(0, "SAMPLE_ID", sample_id)
        frames.append(sample_df)

    output_df = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 65}")
    print("Reconstruction Complete")
    print(f"{'=' * 65}")
    print(f"  Samples:    {len(reconstructed)}")
    total_steps = sum(arr.shape[0] for arr in reconstructed.values())
    print(f"  Total rows: {total_steps}")
    print(f"  Features:   {n_features}")
    print(f"  Output:     {output_path}")
    print(f"{'=' * 65}\n")


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


@data.command(name="describe", help="Inspect and describe dataset statistics (supports both H5 and CSV files)")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input H5 or CSV file path")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output in JSON format (for machine consumption)")
def data_describe(input_path, json_output):
    """Inspect and describe dataset statistics. Supports both H5 and CSV files."""
    import numpy as np

    from ..utils.logging import logger

    input_ext = os.path.splitext(input_path)[1].lower()

    if input_ext == ".csv":
        _describe_csv(input_path, json_output, logger)
    elif input_ext in (".h5", ".hdf5"):
        _describe_h5(input_path, json_output, logger)
    else:
        raise click.BadParameter(
            f"Unsupported file format '{input_ext}'. Supported: .csv, .h5, .hdf5"
        )


def _describe_csv(input_path, json_output, logger):
    """Describe a CSV file following the ai4ts data protocol."""
    import json as json_mod

    import numpy as np
    import pandas as pd

    logger.info(f"Describing CSV dataset: {input_path}")

    df = pd.read_csv(input_path)
    sample_id_col, label_col, feature_cols = _detect_columns(df)

    if not feature_cols:
        raise click.BadParameter(f"No numeric feature columns found in {input_path}.")

    # compute stats
    if sample_id_col and sample_id_col in df.columns:
        groups = df.groupby(sample_id_col)
        n_samples = groups.ngroups
        steps_per_sample = groups.size()
        n_steps_min = int(steps_per_sample.min())
        n_steps_max = int(steps_per_sample.max())
        n_steps = n_steps_max  # pad to max
        uniform_length = n_steps_min == n_steps_max
    else:
        n_samples = 1
        n_steps = len(df)
        n_steps_min = n_steps_max = n_steps
        uniform_length = True

    n_features = len(feature_cols)
    feature_data = df[feature_cols]
    total_elements = feature_data.size
    nan_count = int(feature_data.isna().sum().sum())
    missing_rate = nan_count / total_elements if total_elements > 0 else 0.0

    # per-feature missing rates
    per_feature_missing = {col: float(feature_data[col].isna().mean()) for col in feature_cols}

    n_classes = None
    if label_col:
        y = df[label_col].dropna().unique()
        n_classes = len(y)

    result = {
        "file": input_path,
        "format": "csv",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "n_samples": n_samples,
        "n_steps": n_steps,
        "n_features": n_features,
        "missing_rate": round(missing_rate, 6),
        "uniform_length": uniform_length,
        "has_sample_id": sample_id_col is not None,
        "has_labels": label_col is not None,
        "label_column": label_col,
        "n_classes": n_classes,
        "feature_columns": feature_cols,
        "per_feature_missing_rate": per_feature_missing,
    }

    if not uniform_length:
        result["n_steps_min"] = n_steps_min
        result["n_steps_max"] = n_steps_max

    if json_output:
        print(json_mod.dumps(result, indent=2))
    else:
        print(f"\n{'=' * 65}")
        print(f"Dataset: {input_path} (CSV)")
        print(f"{'=' * 65}")
        print(f"  Total rows:     {len(df)}")
        print(f"  Total columns:  {len(df.columns)}")
        print(f"  Samples:        {n_samples}")
        print(f"  Time steps:     {n_steps}" + ("" if uniform_length else f" (range: {n_steps_min}-{n_steps_max})"))
        print(f"  Features:       {n_features}")
        print(f"  Missing rate:   {missing_rate:.2%} ({nan_count}/{total_elements})")
        print(f"  SAMPLE_ID:      {'yes' if sample_id_col else 'no (single sample)'}")
        if label_col:
            print(f"  Labels:         yes — column '{label_col}' ({n_classes} classes)")
        else:
            print(f"  Labels:         no")

        print(f"\n  Feature columns: {feature_cols}")
        print(f"\n  Per-feature missing rates:")
        for col, rate in per_feature_missing.items():
            bar = "█" * int(rate * 30) + "░" * (30 - int(rate * 30))
            print(f"    {col:20s} {bar} {rate:.2%}")

        print(f"\n{'=' * 65}")
        print("To prepare for PyPOTS:")
        print(f"  pypots-cli data prepare --input {input_path} --output data.h5 --task <task>")
        print(f"{'=' * 65}\n")


def _describe_h5(input_path, json_output, logger):
    """Describe an H5 file."""
    import json as json_mod

    import numpy as np

    from ..data.saving.h5 import load_dict_from_h5

    logger.info(f"Describing H5 dataset: {input_path}")

    loaded = load_dict_from_h5(input_path)

    n_samples = None
    n_features = None
    seq_length = None
    keys_info = {}

    for key, value in loaded.items():
        if isinstance(value, np.ndarray):
            info = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            if np.issubdtype(value.dtype, np.number):
                total_elements = value.size
                nan_count = int(np.isnan(value).sum()) if np.issubdtype(value.dtype, np.floating) else 0
                info["missing_rate"] = round(nan_count / total_elements, 6) if total_elements > 0 else 0.0
                info["min"] = float(np.nanmin(value))
                info["max"] = float(np.nanmax(value))
                info["mean"] = float(np.nanmean(value))
                info["std"] = float(np.nanstd(value))

            if key == "X":
                n_samples = value.shape[0]
                if value.ndim >= 3:
                    seq_length = value.shape[1]
                    n_features = value.shape[2]
                elif value.ndim == 2:
                    n_features = value.shape[1]

            keys_info[key] = info
        else:
            keys_info[key] = {"type": type(value).__name__, "value": str(value)}

    result = {
        "file": input_path,
        "format": "h5",
        "n_keys": len(loaded),
        "keys": keys_info,
        "n_samples": n_samples,
        "n_steps": seq_length,
        "n_features": n_features,
    }

    if json_output:
        print(json_mod.dumps(result, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {input_path}")
        print(f"{'=' * 60}")
        print(f"Number of keys: {len(loaded)}")
        print(f"{'-' * 60}")

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
