try:
    from IPython.html.widgets import interactive, FloatSliderWidget
    from IPython.display import Math, display
except ImportError:
    pass

from ..small_scripts import unitstr


def widget(gp, outputfn=None, ranges=None):
    gp = gp.copy()

    if not ranges:
        ranges = {k._cmpstr: (min(vs), max(vs), (max(vs)-min(vs))/100.0)
                  for k, vs in gp.sweep.items()}
        ranges.update({k._cmpstr: FloatSliderWidget(min=v/10.0, max=10*v , step=v/10.0, value=v)
                       for k, v in gp.substitutions.items()})
    if not outputfn:
        def outputfn(gp):
            print gp.solution.table(["cost", "free_variables"])

    gp.sweep = {}
    gp.prewidget = gp.last

    def display(**subs):
        gp.sub(subs, replace=True)
        if hasattr(gp, "localsolve"):
            gp.localsolve(printing=False)
        else:
            gp.solve(printing=False)
        outputfn(gp)
        gp.load(gp.prewidget)

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
