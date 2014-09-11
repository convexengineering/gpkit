"""Tools for setting up printing in interactive sessions. """

from __future__ import print_function, division

from io import BytesIO

from sympy import latex as default_latex
from sympy import preview
from sympy.core.compatibility import integer_types
from sympy.utilities.misc import debug


def _init_python_printing(stringify_func):
    """Setup printing in Python interactive session. """
    import sys
    from sympy.core.compatibility import builtins

    def _displayhook(arg):
        """Python's pretty-printer display hook.

This function was adapted from:

http://www.python.org/dev/peps/pep-0217/

"""
        if arg is not None:
            builtins._ = None
            print(stringify_func(arg))
            builtins._ = arg

    sys.displayhook = _displayhook


def _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor,
                           backcolor, fontsize, latex_mode, print_builtin,
                           latex_printer):
    """Setup printing in IPython interactive session. """
    try:
        from IPython.lib.latextools import latex_to_png
    except ImportError:
        pass

    preamble = "\\documentclass[%s]{article}\n" \
               "\\pagestyle{empty}\n" \
               "\\usepackage{amsmath,amsfonts}%s\\begin{document}"
    if euler:
        addpackages = '\\usepackage{euler}'
    else:
        addpackages = ''
    preamble = preamble % (fontsize, addpackages)

    imagesize = 'tight'
    offset = "0cm,0cm"
    resolution = 150
    dvi = r"-T %s -D %d -bg %s -fg %s -O %s" % (
        imagesize, resolution, backcolor, forecolor, offset)
    dvioptions = dvi.split()
    debug("init_printing: DVIOPTIONS:", dvioptions)
    debug("init_printing: PREAMBLE:", preamble)

    latex = latex_printer or default_latex

    def _print_plain(arg, p, cycle):
        """caller for pretty, for use in IPython 0.11"""
        if _can_print_latex(arg):
            p.text(stringify_func(arg))
        else:
            p.text(IPython.lib.pretty.pretty(arg))

    def _preview_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='png', viewer='BytesIO',
                    outputbuffer=exprbuffer, preamble=preamble,
                    dvioptions=dvioptions)
        except Exception as e:
            # IPython swallows exceptions
            debug("png printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        return exprbuffer.getvalue()

    def _matplotlib_wrapper(o):
        # mathtext does not understand certain latex flags, so we try to
        # replace them with suitable subs
        o = o.replace(r'\operatorname', '')
        o = o.replace(r'\overline', r'\bar')
        return latex_to_png(o)

    def _can_print_latex(o):
        """Return True if type o can be printed with LaTeX.

If o is a container type, this is True if and only if every element of
o can be printed with LaTeX.
"""
        from sympy import Basic
        from sympy.matrices import MatrixBase
        from sympy.physics.vector import Vector, Dyadic
        if isinstance(o, (list, tuple, set, frozenset)):
            return all(_can_print_latex(i) for i in o)
        elif isinstance(o, dict):
            return all(_can_print_latex(i) and _can_print_latex(o[i]) for i in o)
        elif isinstance(o, bool):
            return False
        elif hasattr(o, '_latex'):
            return True
        elif isinstance(o, (float, integer_types)) and print_builtin:
            return True
        return False

    def _print_latex_png(o):
        """
A function that returns a png rendered by an external latex
distribution, falling back to matplotlib rendering
"""
        if _can_print_latex(o):
            s = latex(o, mode=latex_mode)
            try:
                return _preview_wrapper(s)
            except RuntimeError:
                if latex_mode != 'inline':
                    s = latex(o, mode='inline')
                return _matplotlib_wrapper(s)

    def _print_latex_matplotlib(o):
        """
A function that returns a png rendered by mathtext
"""
        if _can_print_latex(o):
            s = latex(o, mode='inline')
            try:
                return _matplotlib_wrapper(s)
            except Exception:
                # Matplotlib.mathtext cannot render some things (like
                # matrices)
                return None

    def _print_latex_text(o):
        """
A function to generate the latex representation of sympy expressions.
"""
        if _can_print_latex(o):
            s = latex(o, mode='plain')
            s = s.replace(r'\dag', r'\dagger')
            s = s.strip('$')
            return '$$%s$$' % s

    def _result_display(self, arg):
        """IPython's pretty-printer display hook, for use in IPython 0.10

This function was adapted from:

ipython/IPython/hooks.py:155

"""
        if self.rc.pprint:
            out = stringify_func(arg)

            if '\n' in out:
                print

            print(out)
        else:
            print(repr(arg))

    import IPython
    if IPython.__version__ >= '0.11':
        from gpkit import Posynomial, array
        printable_types = [Posynomial, array, float, tuple, list, set,
                frozenset, dict] + list(integer_types)

        plaintext_formatter = ip.display_formatter.formatters['text/plain']

        for cls in printable_types:
            plaintext_formatter.for_type(cls, _print_plain)

        png_formatter = ip.display_formatter.formatters['image/png']
        if use_latex in (True, 'png'):
            debug("init_printing: using png formatter")
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_png)
        elif use_latex == 'matplotlib':
            debug("init_printing: using matplotlib formatter")
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_matplotlib)
        else:
            debug("init_printing: not using any png formatter")
            for cls in printable_types:
                # Better way to set this, but currently does not work in IPython
                #png_formatter.for_type(cls, None)
                if cls in png_formatter.type_printers:
                    png_formatter.type_printers.pop(cls)

        latex_formatter = ip.display_formatter.formatters['text/latex']
        if use_latex in (True, 'mathjax'):
            debug("init_printing: using mathjax formatter")
            for cls in printable_types:
                latex_formatter.for_type(cls, _print_latex_text)
        else:
            debug("init_printing: not using text/latex formatter")
            for cls in printable_types:
                # Better way to set this, but currently does not work in IPython
                #latex_formatter.for_type(cls, None)
                if cls in latex_formatter.type_printers:
                    latex_formatter.type_printers.pop(cls)

    else:
        ip.set_hook('result_display', _result_display)


def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor='Black',
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='equation*', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None):
    """
Initializes pretty-printer depending on the environment.

Parameters
==========

pretty_print: boolean
If True, use pretty_print to stringify or the provided pretty
printer; if False, use sstrrepr to stringify or the provided string
printer.
order: string or None
There are a few different settings for this parameter:
lex (default), which is lexographic order;
grlex, which is graded lexographic order;
grevlex, which is reversed graded lexographic order;
old, which is used for compatibility reasons and for long expressions;
None, which sets it to lex.
use_unicode: boolean or None
If True, use unicode characters;
if False, do not use unicode characters.
use_latex: string, boolean, or None
If True, use default latex rendering in GUI interfaces (png and
mathjax);
if False, do not use latex rendering;
if 'png', enable latex rendering with an external latex compiler,
falling back to matplotlib if external compilation fails;
if 'matplotlib', enable latex rendering with matplotlib;
if 'mathjax', enable latex text generation, for example MathJax
rendering in IPython notebook or text rendering in LaTeX documents
wrap_line: boolean
If True, lines will wrap at the end; if False, they will not wrap
but continue as one line. This is only relevant if `pretty_print` is
True.
num_columns: int or None
If int, number of columns before wrapping is set to num_columns; if
None, number of columns before wrapping is set to terminal width.
This is only relevant if `pretty_print` is True.
no_global: boolean
If True, the settings become system wide;
if False, use just for this console/session.
ip: An interactive console
This can either be an instance of IPython,
or a class that derives from code.InteractiveConsole.
euler: boolean, optional, default=False
Loads the euler package in the LaTeX preamble for handwritten style
fonts (http://www.ctan.org/pkg/euler).
forecolor: string, optional, default='Black'
DVI setting for foreground color.
backcolor: string, optional, default='Transparent'
DVI setting for background color.
fontsize: string, optional, default='10pt'
A font size to pass to the LaTeX documentclass function in the
preamble.
latex_mode: string, optional, default='equation*'
The mode used in the LaTeX printer. Can be one of:
{'inline'|'plain'|'equation'|'equation*'}.
print_builtin: boolean, optional, default=True
If true then floats and integers will be printed. If false the
printer will only print SymPy types.
str_printer: function, optional, default=None
A custom string printer function. This should mimic
sympy.printing.sstrrepr().
pretty_printer: function, optional, default=None
A custom pretty printer. This should mimic sympy.printing.pretty().
latex_printer: function, optional, default=None
A custom LaTeX printer. This should mimic sympy.printing.latex()
This should mimic sympy.printing.latex().

Examples
========

>>> from sympy.interactive import init_printing
>>> from sympy import Symbol, sqrt
>>> from sympy.abc import x, y
>>> sqrt(5)
sqrt(5)
>>> init_printing(pretty_print=True) # doctest: +SKIP
>>> sqrt(5) # doctest: +SKIP
___
\/ 5
>>> theta = Symbol('theta') # doctest: +SKIP
>>> init_printing(use_unicode=True) # doctest: +SKIP
>>> theta # doctest: +SKIP
\u03b8
>>> init_printing(use_unicode=False) # doctest: +SKIP
>>> theta # doctest: +SKIP
theta
>>> init_printing(order='lex') # doctest: +SKIP
>>> str(y + x + y**2 + x**2) # doctest: +SKIP
x**2 + x + y**2 + y
>>> init_printing(order='grlex') # doctest: +SKIP
>>> str(y + x + y**2 + x**2) # doctest: +SKIP
x**2 + x + y**2 + y
>>> init_printing(order='grevlex') # doctest: +SKIP
>>> str(y * x**2 + x * y**2) # doctest: +SKIP
x**2*y + x*y**2
>>> init_printing(order='old') # doctest: +SKIP
>>> str(x**2 + y**2 + x + y) # doctest: +SKIP
x**2 + x + y**2 + y
>>> init_printing(num_columns=10) # doctest: +SKIP
>>> x**2 + x + y**2 + y # doctest: +SKIP
x + y +
x**2 + y**2
"""
    import sys
    from sympy.printing.printer import Printer

    if pretty_print:
        if pretty_printer is not None:
            stringify_func = pretty_printer
        else:
            from sympy.printing import pretty as stringify_func
    else:
        if str_printer is not None:
            stringify_func = str_printer
        else:
            from sympy.printing import sstrrepr as stringify_func

    # Even if ip is not passed, double check that not in IPython shell
    if ip is None:
        try:
            ip = get_ipython()
        except NameError:
            pass

    if ip and ip.__module__.startswith('IPython') and pretty_print:
        try:
            import IPython
            # IPython 1.0 deprecates the frontend module, so we import directly
            # from the terminal module to prevent a deprecation message from being
            # shown.
            if IPython.__version__ >= '1.0':
                from IPython.terminal.interactiveshell import TerminalInteractiveShell
            else:
                from IPython.frontend.terminal.interactiveshell import TerminalInteractiveShell
            from code import InteractiveConsole
        except ImportError:
            pass
        else:
            # This will be True if we are in the qtconsole or notebook
            if not isinstance(ip, (InteractiveConsole, TerminalInteractiveShell)) \
                    and 'ipython-console' not in ''.join(sys.argv):
                if use_unicode is None:
                    debug("init_printing: Setting use_unicode to True")
                    use_unicode = True
                if use_latex is None:
                    debug("init_printing: Setting use_latex to True")
                    use_latex = True

    if not no_global:
        Printer.set_global_settings(order=order, use_unicode=use_unicode,
                                    wrap_line=wrap_line, num_columns=num_columns)
    else:
        _stringify_func = stringify_func

        if pretty_print:
            stringify_func = lambda expr: \
                             _stringify_func(expr, order=order,
                                             use_unicode=use_unicode,
                                             wrap_line=wrap_line,
                                             num_columns=num_columns)
        else:
            stringify_func = lambda expr: _stringify_func(expr, order=order)

    if ip is not None and ip.__module__.startswith('IPython'):
        _init_ipython_printing(ip, stringify_func, use_latex, euler,
                               forecolor, backcolor, fontsize, latex_mode,
                               print_builtin, latex_printer)
    else:
        _init_python_printing(stringify_func)