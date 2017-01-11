"Implements plot_sweep1d function"
import matplotlib.pyplot as plt
import numpy as np
from ..exceptions import InvalidGPConstraint
from ..tools import autosweep_1d


def plot_sweep1d(model, swept, swept_over, posy):
    """Creates and plots a sweep from an existing model

    Example usage:
    f, _ = plot_sweep_1d(m, {'x': np.linspace(1, 2, 5)}, 'y')
    f.savefig('mysweep.png')
    """
    f, ax = plt.subplots()
    osub = model.substitutions[swept] if swept in model.substitutions else None

    if isinstance(swept_over, tuple) and len(swept_over) == 3:
        start, end, tol = swept_over
        sol = autosweep_1d(model, tol, swept, [start, end])
        x = np.linspace(start, end, 100)
        if posy == model.cost:
            # should look like a line plot
            ax.fill_between(x, sol["cost"].lb(x), sol["cost"].ub(x),
                            edgecolor="blue", linewidth=0.75)
        else:
            ax.plot(x, sol[posy](x))
    else:
        model.substitutions.update({swept: ('sweep', swept_over)})
        try:
            sol = model.solve()
        except InvalidGPConstraint:
            sol = model.localsolve()
        ax.plot(swept_over, sol(posy))
    xlabel = (swept.key.descr.get("label", swept.key.name)
              + " " + swept.key.unitstr())
    if hasattr(posy, "key"):
        ylabel = (posy.key.descr.get("label", posy.key.name)
                  + " " + posy.key.unitstr())
    else:
        ylabel = str(posy)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if osub:
        model.substitutions[swept] = osub

    return f, ax
