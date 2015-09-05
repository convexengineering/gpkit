cp gpkit/env/settings .
sed -i '1s/.*/installed_solvers : mosek_cli/' gpkit/env/settings
cat gpkit/env/settings
python3 -c "import gpkit.tests; gpkit.tests.helpers.run_tests(gpkit.tests.t_examples.TESTS)"
mv settings gpkit/env
