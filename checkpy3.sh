cp gpkit/env/settings .
sed -i '1s/.*/installed_solvers : mosek_cli/' gpkit/env/settings
cat gpkit/env/settings
python3 docs/source/examples/simpleflight.py
mv settings gpkit/env
