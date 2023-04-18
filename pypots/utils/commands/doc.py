"""
CLI tools to help the development team build PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
import shutil
from argparse import ArgumentParser, Namespace

from pypots.utils.commands import BaseCommand
from pypots.utils.logging import logger
from tsdb.data_processing import _download_and_extract

CLONED_LATEST_PYPOTS = "temp_pypots_latest"

# These files need to be updated while generating the new documentation
DOC_RST_FILES = [
    "pypots.rst",
    "pypots.imputation.rst",
    "pypots.classification.rst",
    "pypots.clustering.rst",
    "pypots.forecasting.rst",
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


def purge_statics():
    logger.info("Directories _build, _static, and _templates will be deleted if exist")
    shutil.rmtree("_build", ignore_errors=True)
    shutil.rmtree("_static", ignore_errors=True)
    shutil.rmtree("_templates", ignore_errors=True)


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
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "doc", help="CLI tools helping build PyPOTS documentation"
        )

        sub_parser.add_argument(
            "--gene_rst",
            dest="gene_rst",
            action="store_true",
            help="Generate rst (reStructuredText) documentation according to the latest code on Github",
        )
        sub_parser.add_argument(
            "--branch",
            type=str,
            default="main",
            choices=["main", "dev"],
            help="Code on which branch will be used for documentation generating",
        )
        sub_parser.add_argument(
            "--gene_html",
            dest="gene_html",
            action="store_true",
            help="Generate the sphinx documentation into static HTML files",
        )
        sub_parser.add_argument(
            "--view_doc",
            dest="view_doc",
            action="store_true",
            help="Deploy the generated HTML documentation locally for view",
        )
        sub_parser.add_argument(
            "--port",
            type=int,
            default=9075,
            help="Use which port to deploy the web server for doc view",  # 9075 looks like "POTS", so use it as default
        )
        sub_parser.add_argument(
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

    def run(self):
        parent_dir = os.path.join(".", "..")
        containing_figs = "figs" in os.listdir(".")
        pypots_in_the_parent_dir = "pypots" in os.listdir(parent_dir)
        # if currently under dir 'docs', it should have sub-dir 'figs', and 'pypots' should be in the parent dir
        whether_under_dir_docs = containing_figs and pypots_in_the_parent_dir

        # `pypots-cli dev` should only be run under dir 'docs'
        # because we probably will compile the doc and generate HTMLs with command `make`
        assert (
            whether_under_dir_docs
        ), "Command `pypots-cli dev` can only be run under the directory 'docs' in project PyPOTS"

        if self._gene_rst:
            try:
                if os.path.exists(CLONED_LATEST_PYPOTS):
                    logger.info(
                        f"Directory {CLONED_LATEST_PYPOTS} exists, deleting it..."
                    )
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
                shutil.rmtree(
                    f"{CLONED_LATEST_PYPOTS}/pypots/tests", ignore_errors=True
                )

                # Generate the docs according to the cloned code
                logger.info("Generating rst files...")
                os.system(
                    "SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,inherited-members "
                    f"sphinx-apidoc {CLONED_LATEST_PYPOTS} -o {CLONED_LATEST_PYPOTS}/rst"
                )

                # Only save the files we need.
                logger.info("Updating the old documentation...")
                for f_ in DOC_RST_FILES:
                    file_to_copy = f"{CLONED_LATEST_PYPOTS}/rst/{f_}"
                    shutil.copy(file_to_copy, ".")

                # Delete the useless files.
                shutil.rmtree(f"{CLONED_LATEST_PYPOTS}", ignore_errors=True)

            except ImportError:
                raise ImportError(IMPORT_ERROR_MESSAGE)
            except Exception as e:
                raise RuntimeError(e)

        if self._gene_html:
            try:
                logger.info("Generating static HTML files...")
                purge_statics()
                os.system(f"make html")
            except Exception as e:
                raise RuntimeError(e)

        if self._view_doc:
            try:
                assert os.path.exists(
                    "_build/html"
                ), "_build/html does not exists, please run `pypots-cli doc --gene_html` first"
                logger.info("Deploying HTML...")
                os.system(
                    f"python -m http.server {self._port} -d _build/html -b 127.0.0.1"
                )
            except Exception as e:
                raise RuntimeError(e)

        if self._cleanup:
            try:
                logger.info("Purging static files...")
                purge_statics()
                logger.info("Purging finished successfully.")

            except Exception as e:
                raise RuntimeError(e)
