"""
CLI command for data management operations (convert, split, describe).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
from argparse import ArgumentParser, Namespace

import numpy as np

from .base import BaseCommand
from ..utils.logging import logger


def data_command_factory(args: Namespace):
    return DataCommand(
        args.action,
        args.input,
        getattr(args, "output", None),
        getattr(args, "output_dir", None),
        getattr(args, "train_ratio", 0.7),
        getattr(args, "val_ratio", 0.1),
        getattr(args, "test_ratio", 0.2),
        getattr(args, "seed", 2024),
    )


class DataCommand(BaseCommand):
    """CLI tools for data management operations including format conversion, dataset splitting, and inspection.

    Examples
    --------
    $ pypots-cli data convert --input data.csv --output data.h5
    $ pypots-cli data split --input full_data.h5 --output_dir ./splits --train_ratio 0.7 --val_ratio 0.1 --test_ratio 0.2
    $ pypots-cli data describe --input data.h5
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "data",
            help="CLI tools for data management operations (convert, split, describe)",
            allow_abbrev=True,
        )

        sub_parser.add_argument(
            "action",
            type=str,
            choices=["convert", "split", "describe"],
            help="The data management action to perform: convert, split, or describe",
        )
        sub_parser.add_argument(
            "--input",
            dest="input",
            type=str,
            required=True,
            help="Input file path",
        )
        sub_parser.add_argument(
            "--output",
            dest="output",
            type=str,
            required=False,
            default=None,
            help="Output file path (used by 'convert' action)",
        )
        sub_parser.add_argument(
            "--output_dir",
            dest="output_dir",
            type=str,
            required=False,
            default=None,
            help="Output directory for saving splits (used by 'split' action)",
        )
        sub_parser.add_argument(
            "--train_ratio",
            dest="train_ratio",
            type=float,
            required=False,
            default=0.7,
            help="Training set ratio (used by 'split' action, default: 0.7)",
        )
        sub_parser.add_argument(
            "--val_ratio",
            dest="val_ratio",
            type=float,
            required=False,
            default=0.1,
            help="Validation set ratio (used by 'split' action, default: 0.1)",
        )
        sub_parser.add_argument(
            "--test_ratio",
            dest="test_ratio",
            type=float,
            required=False,
            default=0.2,
            help="Test set ratio (used by 'split' action, default: 0.2)",
        )
        sub_parser.add_argument(
            "--seed",
            dest="seed",
            type=int,
            required=False,
            default=2024,
            help="Random seed for reproducible splitting (used by 'split' action, default: 2024)",
        )

        sub_parser.set_defaults(func=data_command_factory)

    def __init__(
        self,
        action: str,
        input_path: str,
        output_path: str = None,
        output_dir: str = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 2024,
    ):
        self._action = action
        self._input_path = input_path
        self._output_path = output_path
        self._output_dir = output_dir
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._seed = seed

    def checkup(self):
        """Run checks on the arguments to avoid incorrect usage."""
        if not os.path.exists(self._input_path):
            raise FileNotFoundError(f"Input file not found: {self._input_path}")

        if self._action == "convert":
            if self._output_path is None:
                raise ValueError("The --output argument is required for the 'convert' action.")
        elif self._action == "split":
            if self._output_dir is None:
                raise ValueError("The --output_dir argument is required for the 'split' action.")
            ratio_sum = self._train_ratio + self._val_ratio + self._test_ratio
            if abs(ratio_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Train/val/test ratios must sum to 1.0, but got {ratio_sum:.6f} "
                    f"({self._train_ratio} + {self._val_ratio} + {self._test_ratio})."
                )

    def run(self):
        """Execute the given data command."""
        self.checkup()

        if self._action == "convert":
            self._run_convert()
        elif self._action == "split":
            self._run_split()
        elif self._action == "describe":
            self._run_describe()

    def _run_convert(self):
        """Convert data between formats."""
        input_ext = os.path.splitext(self._input_path)[1].lower()
        output_ext = os.path.splitext(self._output_path)[1].lower()

        logger.info(f"Converting {self._input_path} ({input_ext}) -> {self._output_path} ({output_ext})")

        # load input data
        if input_ext == ".csv":
            import pandas as pd

            df = pd.read_csv(self._input_path)
            data = {"X": df.values}
            logger.info(f"Loaded CSV with shape {df.shape}")
        elif input_ext == ".npy":
            data = {"X": np.load(self._input_path)}
            logger.info(f"Loaded NumPy array with shape {data['X'].shape}")
        elif input_ext == ".npz":
            data = dict(np.load(self._input_path))
            logger.info(f"Loaded NumPy archive with keys: {list(data.keys())}")
        elif input_ext == ".pkl":
            from ..data.saving.pickle import pickle_load

            data = pickle_load(self._input_path)
            if not isinstance(data, dict):
                data = {"X": data}
            logger.info(f"Loaded Pickle data with keys: {list(data.keys())}")
        else:
            raise ValueError(f"Unsupported input format: {input_ext}. Supported: .csv, .npy, .npz, .pkl")

        # save output data
        if output_ext == ".h5":
            from ..data.saving.h5 import save_dict_into_h5

            save_dict_into_h5(data, self._output_path)
        elif output_ext == ".npy":
            np.save(self._output_path, data["X"])
        elif output_ext == ".pkl":
            from ..data.saving.pickle import pickle_dump

            pickle_dump(data, self._output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_ext}. Supported: .h5, .npy, .pkl")

        logger.info(f"Successfully converted and saved to {self._output_path}")

    def _run_split(self):
        """Split dataset into train/val/test sets."""
        from ..data.saving.h5 import load_dict_from_h5, save_dict_into_h5

        logger.info(
            f"Splitting {self._input_path} with ratios "
            f"train={self._train_ratio}, val={self._val_ratio}, test={self._test_ratio}, seed={self._seed}"
        )

        data = load_dict_from_h5(self._input_path)

        # determine number of samples from the "X" key
        if "X" not in data:
            raise ValueError("The input H5 file must contain an 'X' key to determine the number of samples.")
        n_samples = data["X"].shape[0]
        logger.info(f"Loaded dataset with {n_samples} samples")

        # shuffle indices
        rng = np.random.default_rng(self._seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        # compute split boundaries
        n_train = int(n_samples * self._train_ratio)
        n_val = int(n_samples * self._val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        logger.info(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        # create split dicts by indexing all arrays
        train_data, val_data, test_data = {}, {}, {}
        for key, value in data.items():
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
        os.makedirs(self._output_dir, exist_ok=True)
        save_dict_into_h5(train_data, os.path.join(self._output_dir, "train.h5"))
        save_dict_into_h5(val_data, os.path.join(self._output_dir, "val.h5"))
        save_dict_into_h5(test_data, os.path.join(self._output_dir, "test.h5"))

        logger.info(f"Successfully saved splits to {self._output_dir}/{{train,val,test}}.h5")

    def _run_describe(self):
        """Inspect and describe dataset statistics."""
        from ..data.saving.h5 import load_dict_from_h5

        logger.info(f"Describing dataset: {self._input_path}")

        data = load_dict_from_h5(self._input_path)

        print(f"\n{'=' * 60}")
        print(f"Dataset: {self._input_path}")
        print(f"{'=' * 60}")
        print(f"Number of keys: {len(data)}")
        print(f"{'-' * 60}")

        n_samples = None
        n_features = None
        seq_length = None

        for key, value in data.items():
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
