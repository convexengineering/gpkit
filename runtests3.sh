cp gpkit/env/settings .
sed -i '1s/.*/installed_solvers : mosek_cli/' gpkit/env/settings
cat gpkit/env/settings
python3 -c "import gpkit.tests; gpkit.tests.run(unitless=False)"
mv settings gpkit/env
