from IPython.html.widgets import interactive


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
