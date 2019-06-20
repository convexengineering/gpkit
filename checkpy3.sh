cp gpkit/env/settings .
sed -i '1s/.*/installed_solvers : mosek, cvxopt/' gpkit/env/settings
cat gpkit/env/settings
python3 gpkit/tests/run_tests.py
mv settings gpkit/env
rm *.pkl
rm solution.*
