# try:
from ipywidgets import interactive, FloatSlider
from IPython.display import Math, display
# except ImportError:
#     pass

from ..small_scripts import unitstr


def widget(m, outputfn=None, ranges=None, **solveargs):
    # HACK HACK HACK HACK
    # copy doesn't work, issue 293
    # and this should not use copy anyway
    # keeping for now -- was intended to avoid widgets mutating the GP
    # that should not be possible and needs to be refactored
    original_cost_units = m.cost.units
    # while we're hacking, hack some more -- avoid calling copy
    m = m.__class__(m.cost, m.constraints, m.substitutions)
    m.cost.units = original_cost_units
    # end HACK

    if not ranges:
        ranges = {k._cmpstr: FloatSlider(min=v/10.0, max=10*v,
                                         step=v/10.0, value=v)
                  for k, v in m.constants.items()}
    if not outputfn:
        def outputfn(m):
            print m.solution.table(["cost", "freevariables"])

    solveargs["verbosity"] = 0

    def display(**subs):
        m.substitutions.update(subs)
        m.solve(**solveargs)
        outputfn(m)

    return interactive(display, **ranges)


def table(gp, tablevars, ranges=None):

    def outputfn(gp):
        def nstr(num):
            cstr = "%.4g" % num
            if 'e' in cstr:
                idx = cstr.index('e')
                cstr = "%s\\times 10^{%i}" % (
                       cstr[:idx], int(cstr[idx+1:]))
            return cstr

        def nastr(num_array):
            if len(num_array.shape):
                return ("\\begin{bmatrix}" +
                        " & ".join(nstr(num) for num in num_array) +
                        "\\end{bmatrix}")
            else:
                return nstr(num_array)

        sols = sorted([(var,
                        nastr(val),
                        unitstr(var, "\mathrm{\\left[ %s \\right]}", "L~"),
                        var.descr["label"])
                       for (var, val) in gp.solution["variables"].items()
                       if var in tablevars])
        display(Math("\n".join([r"\begin{array}[rlll]\text{}"]
                     + [r"%s & %s & %s & \text{%s}\\" % sol for sol in sols]
                     + [r"\end{array}"])))

    return widget(gp, outputfn, ranges)
