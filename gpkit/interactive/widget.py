from IPython.html.widgets import interactive
from gpkit.small_scripts import unitstr
from IPython.display import Math, display


def widget(gp, outputfn, ranges):
    gp.prewidget = gp.last

    def display(**kwargs):
        subs = {}
        varkeys = gp.var_locs.keys()
        for key, value in kwargs.items():
            if key in varkeys:
                subs[varkeys[varkeys.index(key)]] = value
        gp.sub(subs)
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

        sols = sorted([(var,
                        nstr(val),
                        unitstr(var, "\mathrm{\\left[ %s \\right]}", "L~"),
                        var.descr["label"])
                       for (var, val) in gp.solution["variables"].items()
                       if var in tablevars])
        display(Math("\n".join([r"\begin{array}[rlll]\text{}"]
                     + [r"%s & %s & %s & \text{%s}\\" % sol for sol in sols]
                     + [r"\end{array}"])))

    return widget(gp, outputfn, sweep)
