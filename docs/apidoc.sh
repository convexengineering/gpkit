rm source/autodoc/*
sphinx-apidoc ../gpkit -o source/autodoc

# Delete fist 3 lines
sed -i '' -e '1,3d' source/autodoc/gpkit.rst

# Add header
header="Glossary\n********\n\n*For an alphabetical listing of all commands, check out the* :ref:\`genindex\`\n"
echo $header | cat - source/autodoc/gpkit.rst > gpkit.rst && mv gpkit.rst source/autodoc/gpkit.rst
