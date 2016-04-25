rm source/autodoc/*
sphinx-apidoc ../gpkit -o source/autodoc
sed -i '1i Glossary\n********\n\n*For an alphabetical listing of all commands, check out the* :ref:`genindex`\n' source/autodoc/gpkit.rst
