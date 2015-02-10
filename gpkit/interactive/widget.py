from IPython.html.widgets import interactive
from IPython.display import Math, display
from ..small_scripts import unitstr


def widget(gp, outputfn, ranges):
    gp.sweep = {}
    gp.prewidget = gp.last

    def display(**kwargs):
        subs = {}
        varkeys = gp.unsubbed.var_locs.keys()
        for key, value in kwargs.items():
            if key in varkeys:
                subs[varkeys[varkeys.index(key)]] = value
        gp.sub(subs, replace=True)
        gp.solve(printing=False)
        outputfn(gp)
        gp.load(gp.prewidget, printing=False)

    return interactive(display, **ranges)


def table(gp, sweep, tablevars):

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

    return widget(gp, outputfn, sweep)
