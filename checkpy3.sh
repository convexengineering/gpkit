cp gpkit/env/settings .
sed -i '1s/.*/installed_solvers : cvxopt/' gpkit/env/settings
cat gpkit/env/settings
#python3 docs/source/examples/simpleflight.py
# python3 gpkit/tests/t_examples.py
python3 gpkit/tests/run_tests.py
mv settings gpkit/env
rm *.pkl
rm solution.*
