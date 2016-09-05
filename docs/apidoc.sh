rm source/autodoc/*
sphinx-apidoc ../gpkit -o source/autodoc

sed -i '1,3d' source/autodoc/gpkit.rst
header="Glossary\n********\n\n*For an alphabetical listing of all commands, check out the* :ref:\`genindex\`\n"
sed -i "1i $header" source/autodoc/gpkit.rst
