from IPython.html.widgets import interactive


def widget(gp, outputfn, ranges):
    gp.prewidget = gp.last
    def display(**kwargs):
        gp.sub(kwargs)
        gp.solve(printing=False)
        outputfn(gp)
        gp.load(gp.prewidget, print_boundwarnings=False)

    return interactive(display, **ranges)
