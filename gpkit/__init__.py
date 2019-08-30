"GP and SP modeling package"
from __future__ import unicode_literals, print_function
__version__ = "0.9.1"

from .build import build
from ._pint import units, ureg, DimensionalityError
from .globals import settings, SignomialsEnabled, Vectorize, NamedVariables
from .varkey import VarKey
from .nomials import Monomial, Posynomial, Signomial, NomialArray
from .nomials import VectorizableVariable as Variable
# NOTE above: the Variable the user sees is not the Variable used internally
from .nomials import VectorVariable, ArrayVariable
from .constraints.gp import GeometricProgram
from .constraints.sgp import SequentialGeometricProgram
from .constraints.sigeq import SignomialEquality
from .constraints.set import ConstraintSet
from .constraints.model import Model
from .tools.docstring import parse_varstring

import inspect
import ast


class parse_variables(object):

    def __init__(self, string, globals):
        self.string = string
        self.globals = globals

    def __call__(self, f):
        string = self.string
        orig_lines, lineno = inspect.getsourcelines(f)
        indent_length = 0
        while orig_lines[1][indent_length] in [" ", "\t"]:
            indent_length += 1
        first_indent_length = indent_length
        next_indented_idx = 2
        # get the next indented line
        while len(orig_lines[next_indented_idx]) <= indent_length + 1:
            next_indented_idx += 1
        while orig_lines[next_indented_idx][indent_length] in [" ", "\t"]:
            indent_length += 1
        second_indent = orig_lines[next_indented_idx][:indent_length]
        parse_lines = [second_indent + line + "\n"
                       for line in parse_varstring(string, errorcatch=False).split("\n")]
        # make ast of these new lines, insert it into the original ast
        new_lines = [orig_lines[1]] + parse_lines + orig_lines[2:]
        # print(new_lines)
        new_src = "\n".join([line[first_indent_length:-1] for line in new_lines])
        # print("ns\n%s" % new_src)
        # print(lineno, new_src)
        new_ast = ast.parse(new_src, "<parse_variables>")
        # ast.fix_missing_locations(new_ast)
        # print(len(parse_lines), parse_lines)
        ast.increment_lineno(new_ast, n=lineno-len(parse_lines))
        code = compile(new_ast, inspect.getsourcefile(f), "exec", dont_inherit=True)
        out = {}
        exec(code, self.globals, out)
        return out[f.__name__]


GPBLU = "#59ade4"
GPCOLORS = ["#59ade4", "#FA3333"]

if "just built!" in settings:
    from .tests.run_tests import run
    run(verbosity=1)
    print("""
GPkit is now installed with solver(s) %s
To incorporate new solvers at a later date, run `gpkit.build()`.

If any tests didn't pass, please post the output above
(starting from "Found no installed solvers, beginning a build.")
to gpkit@mit.edu or https://github.com/convexengineering/gpkit/issues/new
so we can prevent others from having these errors.

The same goes for any other bugs you encounter with GPkit:
send 'em our way, along with any interesting models, speculative features,
comments, discussions, or clarifications you feel like sharing.

Finally, we hope you find our documentation (https://gpkit.readthedocs.io/)
and engineering-design models (https://github.com/convexengineering/gplibrary/)
to be useful resources for your own applications.

Enjoy!
""" % settings["installed_solvers"])
