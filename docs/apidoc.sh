rm source/autodoc/*
sphinx-apidoc ../gpkit -o source/autodoc

# Delete first 3 lines
tail -n+3 source/autodoc/gpkit.rst

# Add header
header="Glossary\n********\n\n*For an alphabetical listing of all commands, check out the* :ref:\`genindex\`\n"
echo $header | cat - source/autodoc/gpkit.rst > gpkit.rst && mv gpkit.rst source/autodoc/gpkit.rst
