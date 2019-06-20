pylint --rcfile .pylintrc gpkit | grep "Raw metrics" -A 14 | tail -n 13
echo "Just tests"
pylint --rcfile .pylintrc gpkit/tests | grep "Raw metrics" -A 14 | tail -n 13
