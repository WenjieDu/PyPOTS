pytest
rm -rf .pytest_cache
black .
flake8 .