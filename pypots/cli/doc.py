"""
CLI tools to help the development team build PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import shutil

import click

from .base import execute_command, check_if_under_root_dir

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


def purge_temp_files():
    from ..utils.logging import logger

    logger.info(f"Directories _build and {CLONED_LATEST_PYPOTS} will be deleted if exist")
    shutil.rmtree("docs/_build", ignore_errors=True)
    shutil.rmtree(CLONED_LATEST_PYPOTS, ignore_errors=True)


@click.command(name="doc", help="CLI tools helping build PyPOTS documentation")
@click.option(
    "--gene-rst",
    "--gene_rst",
    is_flag=True,
    help="Generate rst (reStructuredText) documentation according to the latest code on Github",
)
@click.option(
    "-b",
    "--branch",
    type=click.Choice(["main", "dev"]),
    default="main",
    help="Code on which branch will be used for documentation generating",
)
@click.option(
    "--gene-html",
    "--gene_html",
    is_flag=True,
    help="Generate the sphinx documentation into static HTML files",
)
@click.option(
    "--view-doc",
    "--view_doc",
    is_flag=True,
    help="Deploy the generated HTML documentation locally for view",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=9075,
    help="Use which port to deploy the web server for doc view",
)
@click.option(
    "-c",
    "--cleanup",
    is_flag=True,
    help="Delete all caches and static resources like HTML and CSS files",
)
def doc(gene_rst, branch, gene_html, view_doc, port, cleanup):
    """Execute the doc command."""
    from ..utils.logging import logger

    # run checks
    check_if_under_root_dir(strict=True)

    if cleanup:
        assert not gene_rst and not gene_html and not view_doc, (
            "Argument `--cleanup` should be used alone. Try `pypots-cli doc --cleanup`"
        )

    try:
        if cleanup:
            logger.info("Purging static files...")
            purge_temp_files()
            logger.info("Purging finished successfully.")

        if gene_rst:
            if os.path.exists(CLONED_LATEST_PYPOTS):
                logger.info(f"Directory {CLONED_LATEST_PYPOTS} exists, deleting it...")
                shutil.rmtree(CLONED_LATEST_PYPOTS, ignore_errors=True)

            # Download the latest code from GitHub
            logger.info(
                f"Downloading PyPOTS with the latest code on branch '{branch}' "
                f"from GitHub into {CLONED_LATEST_PYPOTS}..."
            )
            url = f"https://github.com/WenjieDu/PyPOTS/archive/refs/heads/{branch}.zip"
            from tsdb.utils.downloading import _download_and_extract

            _download_and_extract(url=url, saving_path=CLONED_LATEST_PYPOTS)

            code_dir = f"{CLONED_LATEST_PYPOTS}/PyPOTS-{branch}"
            files_to_move = os.listdir(code_dir)
            destination_dir = os.path.join(os.getcwd(), CLONED_LATEST_PYPOTS)
            for f_ in files_to_move:
                shutil.move(os.path.join(code_dir, f_), destination_dir)
            # delete code in tests because we don't need its doc
            shutil.rmtree(f"{CLONED_LATEST_PYPOTS}/pypots/tests", ignore_errors=True)

            # Generate the docs according to the cloned code
            logger.info("Generating rst files...")
            os.environ["SPHINX_APIDOC_OPTIONS"] = "members,undoc-members,show-inheritance,inherited-members"
            execute_command(f"sphinx-apidoc {CLONED_LATEST_PYPOTS} -o {CLONED_LATEST_PYPOTS}/rst")

            # Only save the files we need.
            logger.info("Updating the old documentation...")
            for f_ in DOC_RST_FILES:
                file_to_copy = f"{CLONED_LATEST_PYPOTS}/rst/{f_}"
                shutil.copy(file_to_copy, "docs")

            # Delete the useless files.
            shutil.rmtree(f"{CLONED_LATEST_PYPOTS}", ignore_errors=True)

        if gene_html:
            logger.info("Generating static HTML files...")
            purge_temp_files()
            execute_command("cd docs && make html")

        if view_doc:
            assert os.path.exists("docs/_build/html"), (
                "docs/_build/html does not exists, please run `pypots-cli doc --gene_html` first"
            )
            logger.info(f"Deploying HTML to http://127.0.0.1:{port}...")
            execute_command(f"python -m http.server {port} -d docs/_build/html -b 127.0.0.1")

    except ImportError:
        raise ImportError(IMPORT_ERROR_MESSAGE)
    except Exception as e:
        raise RuntimeError(e)
