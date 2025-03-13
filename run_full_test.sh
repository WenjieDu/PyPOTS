push_flag=false
coverage_filename="coverage.lcov"

for arg in "$@"; do
  if [[ "$arg" == "--push" ]]; then
    push_flag=true
  fi
done

# clean up and clone the full test branch of PyPOTS
rm -rf full_test_pypots
git clone https://github.com/WenjieDu/PyPOTS full_test_pypots -b full_test --depth 1
cd full_test_pypots

# run full testing
python tests/global_test_config.py
python -m pytest -rA tests/*/* -s -n 1 --cov=pypots --dist=loadgroup --cov-config=.coveragerc
ENABLE_AMP=1 python -m pytest -rA tests/imputation/llms/* -s -n 1 --cov=pypots --dist=loadgroup --cov-config=.coveragerc --cov-append
ENABLE_AMP=1 python -m pytest -rA tests/forecasting/llms/* -s -n 1 --cov=pypots --dist=loadgroup --cov-config=.coveragerc --cov-append

# if push flag is given, push the coverage report to coveralls
if [[ "$push_flag" == true ]]; then
  python -m coverage lcov

  # check if the coverage report exists
  if [ -f "$coverage_filename" ]; then
    coveralls
  else
    echo "$coverage_filename does not exist. No data pushed to coveralls."
  fi

fi

cd ..