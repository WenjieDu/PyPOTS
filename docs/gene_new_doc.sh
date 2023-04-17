# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

# This shell script is created to help generate PyPOTS documentation.


git clone --depth=1 https://github.com/WenjieDu/PyPOTS pypots_latest && rm -rf pypots_latest/pypots/tests

# Generate the docs according to the latest code on branch main
SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,inherited-members sphinx-apidoc pypots_latest -o pypots_latest/rst

# Only save the files we need.
cp pypots_latest/rst/pypots.classification.rst \
  pypots_latest/rst/pypots.clustering.rst \
  pypots_latest/rst/pypots.data.rst \
  pypots_latest/rst/pypots.forecasting.rst \
  pypots_latest/rst/pypots.imputation.rst \
  pypots_latest/rst/pypots.rst \
  pypots_latest/rst/pypots.utils.rst \
    .

# Delete the useless files.
rm -rf pypots_latest
