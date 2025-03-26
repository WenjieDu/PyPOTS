"""
CLI tools to help the development team build PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import shutil
from argparse import Namespace

from tsdb.utils.downloading import _download_and_extract

from ..cli.base import BaseCommand
from ..utils.logging import logger

CLONED_LATEST_PYPOTS = "temp_pypots_latest"

# These files need to be updated while generating the new documentation
DOC_RST_FILES = [
    "pypots.rst",
    "pypots.imputation.rst",
    "pypots.classification.rst",
    "pypots.clustering.rst",
    "pypots.forecasting.rst",
    "pypots.anomaly_detection.rst",
    "pypots.representation.rst",
    "pypots.data.rst",
    "pypots.utils.rst",
]

IMPORT_ERROR_MESSAGE = (
    "`pypots-cli doc` command is for PyPOTS developers to build documentation easily. "
    "Therefore, you need a complete PyPOTS development environment. However, you are missing some dependencies. "
    "Please refer to https://github.com/WenjieDu/PyPOTS/blob/main/environment-dev.yml for dependency details. "
)


def doc_command_factory(args: Namespace):
    return DocCommand(
        args.gene_rst,
        args.branch,
        args.gene_html,
        args.view_doc,
        args.port,
        args.cleanup,
    )


def purge_temp_files():
    logger.info(f"Directories _build and {CLONED_LATEST_PYPOTS} will be deleted if exist")
    shutil.rmtree("docs/_build", ignore_errors=True)
    shutil.rmtree(CLONED_LATEST_PYPOTS, ignore_errors=True)


class DocCommand(BaseCommand):
    """CLI tools helping build PyPOTS documentation.

    Notes
    -----
    Only code pushed to GitHub can be used for documentation generation.
    During generation, PyPOTS repo on GitHub will firstly be cloned to local with the branch specified by
    argument `--branch`. The generated rst files will replace the old ones directly. To view the updated
    documentation, use option `--view_doc` to generate the doc into HTMLs and deploy it locally for view.
    After running `--view_doc`, the generated static files won't be deleted automatically. To purge them,
    run the command with option `--cleanup`.

    Examples
    --------
    $ pypots-cli doc --gene_rst [--branch main]
    $ pypots-cli doc --view_doc [--port 9075]
    $ pypots-cli doc --cleanup

    """

    @staticmethod
    def register_subcommand(parser):
        sub_parser = parser.add_parser(
            "doc",
            help="CLI tools helping build PyPOTS documentation",
            allow_abbrev=True,
        )

        sub_parser.add_argument(
            "--gene-rst",
            "--gene_rst",
            dest="gene_rst",
            action="store_true",
            help="Generate rst (reStructuredText) documentation according to the latest code on Github",
        )
        sub_parser.add_argument(
            "-b",
            "--branch",
            type=str,
            default="main",
            choices=["main", "dev"],
            help="Code on which branch will be used for documentation generating",
        )
        sub_parser.add_argument(
            "--gene-html",
            "--gene_html",
            dest="gene_html",
            action="store_true",
            help="Generate the sphinx documentation into static HTML files",
        )
        sub_parser.add_argument(
            "--view-doc",
            "--view_doc",
            dest="view_doc",
            action="store_true",
            help="Deploy the generated HTML documentation locally for view",
        )
        sub_parser.add_argument(
            "-p",
            "--port",
            type=int,
            default=9075,
            help="Use which port to deploy the web server for doc view",  # 9075 looks like "POTS", so use it as default
        )
        sub_parser.add_argument(
            "-c",
            "--cleanup",
            dest="cleanup",
            action="store_true",
            help="Delete all caches and static resources like HTML and CSS files ",
        )

        sub_parser.set_defaults(func=doc_command_factory)

    def __init__(
        self,
        gene_rst: bool,
        branch: str,
        gene_html: bool,
        view_doc: bool,
        port: int,
        cleanup: bool,
    ):
        self._gene_rst = gene_rst
        self._branch = branch
        self._gene_html = gene_html
        self._view_doc = view_doc
        self._port = port
        self._cleanup = cleanup

    def checkup(self):
        """Run some checks on the arguments to avoid error usages"""
        self.check_if_under_root_dir(strict=True)

        if self._cleanup:
            assert (
                not self._gene_rst and not self._gene_html and not self._view_doc
            ), "Argument `--cleanup` should be used alone. Try `pypots-cli doc --cleanup`"

    def run(self):
        """Execute the given command."""
        # run checks first
        self.checkup()

        try:
            if self._cleanup:
                logger.info("Purging static files...")
                purge_temp_files()
                logger.info("Purging finished successfully.")

            if self._gene_rst:
                if os.path.exists(CLONED_LATEST_PYPOTS):
                    logger.info(f"Directory {CLONED_LATEST_PYPOTS} exists, deleting it...")
                    shutil.rmtree(CLONED_LATEST_PYPOTS, ignore_errors=True)

                # Download the latest code from GitHub
                logger.info(
                    f"Downloading PyPOTS with the latest code on branch '{self._branch}' "
                    f"from GitHub into {CLONED_LATEST_PYPOTS}..."
                )
                url = f"https://github.com/WenjieDu/PyPOTS/archive/refs/heads/{self._branch}.zip"
                _download_and_extract(url=url, saving_path=CLONED_LATEST_PYPOTS)

                code_dir = f"{CLONED_LATEST_PYPOTS}/PyPOTS-{self._branch}"
                files_to_move = os.listdir(code_dir)
                destination_dir = os.path.join(os.getcwd(), CLONED_LATEST_PYPOTS)
                for f_ in files_to_move:
                    shutil.move(os.path.join(code_dir, f_), destination_dir)
                # delete code in tests because we don't need its doc
                shutil.rmtree(f"{CLONED_LATEST_PYPOTS}/pypots/tests", ignore_errors=True)

                # Generate the docs according to the cloned code
                logger.info("Generating rst files...")
                os.environ["SPHINX_APIDOC_OPTIONS"] = "members,undoc-members,show-inheritance,inherited-members"
                self.execute_command(f"sphinx-apidoc {CLONED_LATEST_PYPOTS} -o {CLONED_LATEST_PYPOTS}/rst")

                # Only save the files we need.
                logger.info("Updating the old documentation...")
                for f_ in DOC_RST_FILES:
                    file_to_copy = f"{CLONED_LATEST_PYPOTS}/rst/{f_}"
                    shutil.copy(file_to_copy, "docs")

                # Delete the useless files.
                shutil.rmtree(f"{CLONED_LATEST_PYPOTS}", ignore_errors=True)

            if self._gene_html:
                logger.info("Generating static HTML files...")
                purge_temp_files()
                self.execute_command("cd docs && make html")

            if self._view_doc:
                assert os.path.exists(
                    "docs/_build/html"
                ), "docs/_build/html does not exists, please run `pypots-cli doc --gene_html` first"
                logger.info(f"Deploying HTML to http://127.0.0.1:{self._port}...")
                self.execute_command(f"python -m http.server {self._port} -d docs/_build/html -b 127.0.0.1")

        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        except Exception as e:
            raise RuntimeError(e)
