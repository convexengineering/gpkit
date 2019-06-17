cp gpkit/env/settings .
sed -i '1s/.*/installed_solvers : cvxopt, mosek_cli/' gpkit/env/settings
cat gpkit/env/settings
#python3 docs/source/examples/simpleflight.py
python3 gpkit/tests/t_examples.py
mv settings gpkit/env
rm *.pkl
rm solution.*
