"""
CLI tools to help initialize environments for running and developing PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from .utils import load_package_from_path
from ..classification import BRITS as BRITS_classification
from ..classification import Raindrop, GRUD
from ..clustering import CRLI, VaDER
from ..imputation import SAITS, Transformer, CSDI, USGAN, GPVAE, MRNN, BRITS, TimesNet
from ..optim import Adam
from ..utils.logging import logger

try:
    import nni
except ImportError:
    logger.error(
        "Hyperparameter tuning mode needs NNI (https://github.com/microsoft/nni) installed, "
        "but is missing in the current environment."
    )


NN_MODELS = {
    # imputation models
    "pypots.imputation.SAITS": SAITS,
    "pypots.imputation.Transformer": Transformer,
    "pypots.imputation.TimesNet": TimesNet,
    "pypots.imputation.CSDI": CSDI,
    "pypots.imputation.USGAN": USGAN,
    "pypots.imputation.GPVAE": GPVAE,
    "pypots.imputation.BRITS": BRITS,
    "pypots.imputation.MRNN": MRNN,
    # classification models
    "pypots.classification.GRUD": GRUD,
    "pypots.classification.BRITS": BRITS_classification,
    "pypots.classification.Raindrop": Raindrop,
    # clustering models
    "pypots.clustering.CRLI": CRLI,
    "pypots.clustering.VaDER": VaDER,
}


def env_command_factory(args: Namespace):
    return TuningCommand(
        args.model,
        args.model_package_path,
        args.train_set,
        args.val_set,
    )


class TuningCommand(BaseCommand):
    """CLI tools helping users and developer setup python environments for running and developing PyPOTS.

    Notes
    -----
    Using this tool supposes that you've already installed `pypots` with at least the scope of `basic` dependencies.
    Please refer to file setup.cfg in PyPOTS project's root dir for definitions of different dependency scopes.

    Examples
    --------
    $ pypots-cli tuning --model pypots.imputation.SAITS --train_set path_to_the_train_set --val_set path_to_the_val_set

    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "tuning",
            help="CLI tools helping run hyper-parameter tuning for specified models",
            allow_abbrev=True,
        )
        sub_parser.add_argument(
            "--model",
            dest="model",
            type=str,
            required=True,
            help="Install specified dependencies in the current python environment",
        )
        sub_parser.add_argument(
            "--model_package_path",
            dest="model_package_path",
            type=str,
            required=False,
            help="If the model is not in the pypots package, specify the path to the model package here.",
        )
        sub_parser.add_argument(
            "--train_set",
            dest="train_set",
            type=str,
            required=True,
            help="",
        )
        sub_parser.add_argument(
            "--val_set",
            dest="val_set",
            type=str,
            required=True,
            help="",
        )
        sub_parser.set_defaults(func=env_command_factory)

    def __init__(
        self,
        model: str,
        model_package_path: str,
        train_set: str,
        val_set: str,
    ):
        self._model = model
        self._model_package_path = model_package_path
        self._train_set = train_set
        self._val_set = val_set

    def checkup(self):
        """Run some checks on the arguments to avoid error usages"""
        pass

    def run(self):
        """Execute the given command."""
        if os.getenv("enable_tuning", False):
            # fetch a new set of hyperparameters from NNI tuner
            tuner_params = nni.get_next_parameter()
            # get the specified model class
            if self._model not in NN_MODELS:
                logger.info(
                    f"The specified model {self._model} is not in PyPOTS. Available models are {NN_MODELS.keys()}. "
                    f"Trying to fetch it from the given model package {self._model_package_path}."
                )
                assert self._model_package_path is not None, (
                    f"The given model {self._model} is not in PyPOTS. "
                    f"Please give the full import path of the model in PyPOTS like pypots.imputation.SAITS\n"
                    f"If you're trying to tune an outside model, "
                    f"please specify the path to the model package with argument `--model_package_path`."
                )
                model_package = load_package_from_path(self._model_package_path)
                assert self._model in model_package.__all__, (
                    f"{self._model} is not in the given model package {self._model_package_path}."
                    f"Please ensure that the model class is in the __all__ list of the model package."
                )
                model_class = getattr(model_package, self._model)
            else:
                if self._model_package_path is not None:
                    logger.warning(
                        f"‼️ Find the specified model {self._model} in PyPOTS, "
                        f"but also find the argument --model_package_path is not None."
                        f"Note that --model_package_path is ignored."
                    )

                model_class = NN_MODELS[self._model]
            # pop out the learning rate
            lr = tuner_params.pop("lr")

            # check if hyperparameters match
            model_all_arguments = model_class.__init__.__annotations__.keys()
            tuner_params_set = set(tuner_params.keys())
            model_arguments_set = set(model_all_arguments)
            if_hyperparameter_match = tuner_params_set.issubset(model_arguments_set)
            if not if_hyperparameter_match:  # raise runtime error if mismatch
                hyperparameter_intersection = tuner_params_set.intersection(
                    model_arguments_set
                )
                mismatched = tuner_params_set.difference(
                    set(hyperparameter_intersection)
                )
                raise RuntimeError(
                    f"Hyperparameters do not match. Mismatched hyperparameters "
                    f"(in the tuning configuration but not in the given model's arguments): {list(mismatched)}"
                )

            # initializing optimizer and model
            # if tuning a GAN model, we need two optimizers
            if "G_optimizer" in model_all_arguments:
                # optimizer for the generator
                tuner_params["G_optimizer"] = Adam(lr=lr)
                # optimizer for the discriminator
                tuner_params["D_optimizer"] = Adam(lr=lr)
            else:
                tuner_params["optimizer"] = Adam(lr=lr)

            # init an instance with the given hyperparameters for the model class
            model = model_class(**tuner_params)
            # train the model and report to NNI
            model.fit(train_set=self._train_set, val_set=self._val_set)
        else:
            raise RuntimeError("Argument `enable_tuning` is not set. Aborting...")
