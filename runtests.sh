cp gpkit/env/settings .
sed -i '1s/.*/installed_solvers : cvxopt/' gpkit/env/settings
cat gpkit/env/settings
python -c "import gpkit.tests; gpkit.tests.run()"
mv settings gpkit/env
rm *.p
