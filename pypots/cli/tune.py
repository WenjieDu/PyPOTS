"""
CLI command for hyperparameter optimization with NNI.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import json
import os
import tempfile
from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from .utils import (
    SUPPORTED_TASKS,
    load_config,
    merge_config_with_args,
    get_model_class,
    get_model_init_params,
)
from ..utils.logging import logger

# Supported NNI search space types and their expected keys
_SEARCH_SPACE_TYPES = {
    "choice": "values",
    "loguniform": "range",
    "uniform": "range",
    "randint": "range",
    "quniform": "range",
}


def _convert_search_space_to_nni(search_space: dict) -> dict:
    """Convert our YAML search_space format to NNI's format.

    Our format example::

        lr:
          type: loguniform
          range: [0.00001, 0.1]
        d_model:
          type: choice
          values: [32, 64, 128, 256]

    NNI format example::

        {"lr": {"_type": "loguniform", "_value": [0.00001, 0.1]},
         "d_model": {"_type": "choice", "_value": [32, 64, 128, 256]}}

    Parameters
    ----------
    search_space : dict
        Search space in our YAML config format.

    Returns
    -------
    nni_search_space : dict
        Search space in NNI format.
    """
    nni_search_space = {}
    for param_name, param_cfg in search_space.items():
        sp_type = param_cfg.get("type")
        assert sp_type in _SEARCH_SPACE_TYPES, (
            f"Unsupported search space type '{sp_type}' for parameter '{param_name}'. "
            f"Supported types: {list(_SEARCH_SPACE_TYPES.keys())}"
        )

        value_key = _SEARCH_SPACE_TYPES[sp_type]
        assert value_key in param_cfg, (
            f"Search space parameter '{param_name}' with type '{sp_type}' "
            f"requires key '{value_key}', but it was not found. Got keys: {list(param_cfg.keys())}"
        )

        nni_search_space[param_name] = {
            "_type": sp_type,
            "_value": param_cfg[value_key],
        }
    return nni_search_space


def tune_command_factory(args: Namespace):
    return TuneCommand(
        config=args.config,
        task=args.task,
        model=args.model,
        n_trials=args.n_trials,
        device=args.device,
        port=args.port,
    )


class TuneCommand(BaseCommand):
    """CLI command for hyperparameter optimization using NNI.

    This is an improved interface over the low-level ``hpo`` command. It accepts a YAML/JSON
    configuration file that describes the model, search space, tuner, and data paths, then
    either launches NNI programmatically or generates the config files for manual execution.

    Examples
    --------
    $ pypots-cli tune --config tune_config.yaml
    $ pypots-cli tune --config tune_config.yaml --task imputation --model SAITS --n_trials 100
    $ pypots-cli tune --config tune_config.yaml --device cuda:0 --port 8888
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "tune",
            help="Run hyperparameter optimization for a PyPOTS model via NNI",
            allow_abbrev=True,
        )

        sub_parser.add_argument(
            "--config",
            dest="config",
            type=str,
            required=True,
            help="Path to a YAML or JSON tuning configuration file",
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
            "--n_trials",
            dest="n_trials",
            type=int,
            default=None,
            help="Override the maximum number of tuning trials (default: 50)",
        )
        sub_parser.add_argument(
            "--device",
            dest="device",
            type=str,
            default=None,
            help="Override the device to use (e.g. 'cpu', 'cuda:0')",
        )
        sub_parser.add_argument(
            "--port",
            dest="port",
            type=int,
            default=8080,
            help="Port for the NNI web UI (default: 8080)",
        )

        sub_parser.set_defaults(func=tune_command_factory)

    def __init__(
        self,
        config: str,
        task: str = None,
        model: str = None,
        n_trials: int = None,
        device: str = None,
        port: int = 8080,
    ):
        self._config_path = config
        self._task = task
        self._model = model
        self._n_trials = n_trials
        self._device = device
        self._port = port

    def checkup(self):
        """Validate arguments before running."""
        assert os.path.exists(self._config_path), (
            f"Configuration file not found: {self._config_path}"
        )

    def run(self):
        """Execute the hyperparameter optimization pipeline."""
        self.checkup()

        # ------------------------------------------------------------------
        # Step 1: Load configuration from file
        # ------------------------------------------------------------------
        config = load_config(self._config_path)
        logger.info(f"Loaded tuning configuration from '{self._config_path}'")

        # ------------------------------------------------------------------
        # Step 2: Merge CLI overrides into the config
        # ------------------------------------------------------------------
        config = merge_config_with_args(
            config,
            Namespace(task=self._task, device=self._device),
            ["task", "device"],
        )

        # Override n_trials: CLI > tuner section > default 50
        tuner_config = config.get("tuner", {})
        if self._n_trials is not None:
            n_trials = self._n_trials
        else:
            n_trials = tuner_config.get("n_trials", 50)

        # Override model name
        model_config = config.get("model", {})
        if self._model is not None:
            model_name = self._model
        else:
            model_name = model_config.get("name")
        assert model_name is not None, (
            "Model name must be specified in the config file (model.name) or via --model"
        )

        # Resolve task
        task = config.get("task")
        assert task is not None, (
            "Task type must be specified in the config file or via --task"
        )

        # ------------------------------------------------------------------
        # Step 3: Validate model exists in the registry
        # ------------------------------------------------------------------
        logger.info(f"Validating model: task='{task}', model='{model_name}'")
        get_model_class(task, model_name)  # raises if not found

        # ------------------------------------------------------------------
        # Step 4: Validate search_space param names against model __init__
        # ------------------------------------------------------------------
        search_space = config.get("search_space", {})
        assert search_space, "The config file must contain a non-empty 'search_space' section."

        model_params = get_model_init_params(task, model_name)
        model_param_names = set(model_params.keys())
        search_param_names = set(search_space.keys())

        invalid_params = search_param_names - model_param_names
        if invalid_params:
            raise ValueError(
                f"Search space contains parameters not accepted by {model_name}.__init__(): "
                f"{sorted(invalid_params)}. "
                f"Valid parameters: {sorted(model_param_names)}"
            )
        logger.info(f"Search space parameters validated: {sorted(search_param_names)}")

        # ------------------------------------------------------------------
        # Step 5: Convert search_space to NNI format
        # ------------------------------------------------------------------
        nni_search_space = _convert_search_space_to_nni(search_space)
        logger.info(f"Converted search space to NNI format ({len(nni_search_space)} parameters)")

        # ------------------------------------------------------------------
        # Step 6: Resolve remaining config values
        # ------------------------------------------------------------------
        data_config = config.get("data", {})
        train_set = data_config.get("train_set")
        val_set = data_config.get("val_set")
        assert train_set is not None, "data.train_set must be specified in the config file"
        assert val_set is not None, "data.val_set must be specified in the config file"

        device = config.get("device", "cpu")
        seed = config.get("seed", None)
        tuner_name = tuner_config.get("name", "TPE")
        model_key = f"pypots.{task}.{model_name}"

        # Build the trial command that NNI will execute for each trial
        trial_cmd = f"ENABLE_HPO=1 pypots-cli hpo --model {model_key} --train_set {train_set} --val_set {val_set}"
        if seed is not None:
            trial_cmd = f"RANDOM_SEED={seed} {trial_cmd}"

        # ------------------------------------------------------------------
        # Step 7 & 8: Try to launch NNI programmatically, fall back to file generation
        # ------------------------------------------------------------------
        try:
            from nni.experiment import Experiment

            logger.info("NNI is available. Launching experiment programmatically...")
            self._run_nni_experiment(
                Experiment=Experiment,
                nni_search_space=nni_search_space,
                trial_cmd=trial_cmd,
                tuner_name=tuner_name,
                n_trials=n_trials,
                port=self._port,
            )
        except ImportError:
            logger.warning(
                "NNI is not installed. Generating config files for manual execution instead."
            )
            self._generate_nni_configs(
                nni_search_space=nni_search_space,
                trial_cmd=trial_cmd,
                tuner_name=tuner_name,
                n_trials=n_trials,
                port=self._port,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_nni_experiment(
        Experiment,
        nni_search_space: dict,
        trial_cmd: str,
        tuner_name: str,
        n_trials: int,
        port: int,
    ):
        """Launch an NNI experiment programmatically.

        Parameters
        ----------
        Experiment :
            The ``nni.experiment.Experiment`` class.
        nni_search_space : dict
            Search space in NNI format.
        trial_cmd : str
            Shell command NNI executes for each trial.
        tuner_name : str
            Name of the NNI tuner (e.g. "TPE", "Random", "Anneal").
        n_trials : int
            Maximum number of trials.
        port : int
            Port for the NNI web UI.
        """
        experiment = Experiment("local")
        experiment.config.trial_command = trial_cmd
        experiment.config.trial_code_directory = "."
        experiment.config.search_space = nni_search_space
        experiment.config.tuner.name = tuner_name
        experiment.config.max_trial_number = n_trials
        experiment.config.trial_concurrency = 1

        logger.info(f"Starting NNI experiment (tuner={tuner_name}, max_trials={n_trials}, port={port})")
        logger.info(f"Trial command: {trial_cmd}")
        logger.info(f"NNI Web UI will be available at: http://localhost:{port}")

        experiment.run(port)

        # Wait for the experiment to finish
        logger.info("Waiting for NNI experiment to complete... Press Ctrl+C to stop early.")
        try:
            input("Press Enter to stop the experiment and view results...")
        except KeyboardInterrupt:
            pass

        # Print results summary
        try:
            logger.info("=" * 60)
            logger.info("Experiment finished. Retrieving results summary...")
            for trial in experiment.export_data():
                logger.info(f"  Trial {trial.parameter['id']}: {trial.value}")
            logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Could not retrieve experiment results: {e}")
        finally:
            experiment.stop()
            logger.info("NNI experiment stopped.")

    @staticmethod
    def _generate_nni_configs(
        nni_search_space: dict,
        trial_cmd: str,
        tuner_name: str,
        n_trials: int,
        port: int,
    ):
        """Generate NNI config files for manual execution when NNI is not installed.

        Parameters
        ----------
        nni_search_space : dict
            Search space in NNI format.
        trial_cmd : str
            Shell command NNI executes for each trial.
        tuner_name : str
            Name of the NNI tuner.
        n_trials : int
            Maximum number of trials.
        port : int
            Port for the NNI web UI.
        """
        output_dir = os.path.join(os.getcwd(), "nni_generated_configs")
        os.makedirs(output_dir, exist_ok=True)

        # Write search_space.json
        search_space_path = os.path.join(output_dir, "search_space.json")
        with open(search_space_path, "w") as f:
            json.dump(nni_search_space, f, indent=2)

        # Write NNI experiment config YAML
        nni_config = {
            "experimentName": "pypots_tune",
            "trialCommand": trial_cmd,
            "trialCodeDirectory": ".",
            "searchSpaceFile": "search_space.json",
            "trialConcurrency": 1,
            "maxTrialNumber": n_trials,
            "tuner": {"name": tuner_name},
            "trainingService": {"platform": "local"},
        }

        config_path = os.path.join(output_dir, "nni_config.yaml")
        try:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(nni_config, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fall back to JSON if PyYAML is not available
            config_path = os.path.join(output_dir, "nni_config.json")
            with open(config_path, "w") as f:
                json.dump(nni_config, f, indent=2)

        logger.info("=" * 60)
        logger.info("NNI configuration files generated successfully!")
        logger.info(f"  Config file  : {config_path}")
        logger.info(f"  Search space : {search_space_path}")
        logger.info("")
        logger.info("NNI is required for hyperparameter optimization.")
        logger.info("Install it with: pip install nni")
        logger.info("")
        logger.info("Once installed, launch the experiment with:")
        logger.info(f"  nnictl create --config {config_path} --port {port}")
        logger.info("")
        logger.info(f"Or re-run this command to launch programmatically:")
        logger.info(f"  pypots-cli tune --config {os.path.abspath(nni_config.get('trialCodeDirectory', '.'))}/../tune_config.yaml")
        logger.info("=" * 60)
