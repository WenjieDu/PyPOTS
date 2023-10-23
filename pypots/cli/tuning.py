"""
CLI tools to help initialize environments for running and developing PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import os
from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from ..utils.logging import logger
from ..optim import Adam

try:
    import nni
except ImportError:
    logger.error(
        "Hyperparameter tuning mode needs NNI (https://github.com/microsoft/nni) installed, "
        "but is missing in the current environment."
    )


from ..imputation import SAITS, Transformer, CSDI, USGAN, GPVAE, BRITS, MRNN

MODELS = {
    "pypots.imputation.SAITS": SAITS,
    "pypots.imputation.Transformer": Transformer,
    "pypots.imputation.CSDI": CSDI,
    "pypots.imputation.US_GAN": USGAN,
    "pypots.imputation.GP_VAE": GPVAE,
    "pypots.imputation.BRITS": BRITS,
    "pypots.imputation.MRNN": MRNN,
}


def env_command_factory(args: Namespace):
    return TuningCommand(
        args.model,
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
    $ pypots-cli tuning --model pypots.imputation.saits --searching_config SAITS_searching_space.json

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
            choices=[
                "pypots.imputation.SAITS",
                "pypots.imputation.Transformer",
                "pypots.imputation.CSDI",
                "pypots.imputation.US_GAN",
                "pypots.imputation.GP_VAE",
                "pypots.imputation.BRITS",
                "pypots.imputation.MRNN",
            ],
            help="Install specified dependencies in the current python environment",
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
        model: bool,
        train_set: str,
        val_set: str,
    ):
        self._model = model
        self._train_set = train_set
        self._val_set = val_set

    def run(self):
        """Execute the given command."""
        if os.getenv("enable_tuning", False):
            tuner_params = nni.get_next_parameter()
            tuner_params["optimizer"] = Adam(lr=tuner_params["lr"])

            model = MODELS[self._model](**tuner_params)
            model.fit(train_set=self._train_set, val_set=self._val_set)
        else:
            logger.error("Argument `enable_tuning` is not set. Aborting...")
