"Interactive GPkit widgets for iPython notebook"
import ipywidgets as widgets
from traitlets import link
from ..small_scripts import is_sweepvar
from ..small_classes import Numbers
from ..exceptions import InvalidGPConstraint


# pylint: disable=too-many-locals
def modelinteract(model, fns_of_sol, ranges=None, **solvekwargs):
    """Easy model interaction in IPython / Jupyter

    By default, this creates a model with sliders for every constant
    which prints a new solution table whenever the sliders are changed.

    Arguments
    ---------
    fn_of_sol : function
        The function called with the solution after each solve that
        displays the result. By default prints a table.

    ranges : dictionary {str: Slider object or tuple}
        Determines which sliders get created. Tuple values may contain
        two or three floats: two correspond to (min, max), while three
        correspond to (min, step, max)

    **solvekwargs
        kwargs which get passed to the solve()/localsolve() method.
    """
    ranges_out = {}
    if ranges:
        if not isinstance(ranges, dict):
            ranges = {k: None for k in ranges}
        slider_vars = set()
        for k in ranges.keys():
            if k in model.substitutions:  # only if already a constant
                for key in model.varkeys[k]:
                    slider_vars.add(key)
                    ranges[key] = ranges[k]
                del ranges[k]
    else:
        slider_vars = model.substitutions
    for k in slider_vars:
        v = model.substitutions[k]
        if is_sweepvar(v) or isinstance(v, Numbers):
            if is_sweepvar(v):
                # By default, sweep variables become sliders, so we
                # need to to update the model's substitutions such that
                # the first solve (with no substitutions) reflects
                # the substitutions created on slider movement
                _, sweep = v
                v = sweep[0]
                model.substitutions.update({k: v})
            vmin, vmax = v/2.0, v*2.0
            if is_sweepvar(v):
                vmin = min(vmin, min(sweep))
                vmax = max(vmax, min(sweep))
            if ranges and ranges[k]:
                vmin, vmax = ranges[k]
            vstep = (vmax-vmin)/24.0
            varkey_latex = "$"+k.latex(excluded=["models"])+"$"
            floatslider = widgets.FloatSlider(min=vmin, max=vmax,
                                              step=vstep, value=v,
                                              description=varkey_latex)
            floatslider.varkey = k
            ranges_out[str(k)] = floatslider

    if not hasattr(fns_of_sol, "__iter__"):
        fns_of_sol = [fns_of_sol]

    solvekwargs["verbosity"] = 0

    def resolve(**subs):
        "This method gets called each time the user changes something"
        model.substitutions.update(subs)
        try:
            try:
                model.solve(**solvekwargs)
            except InvalidGPConstraint:
                # TypeError raised by as_posyslt1 in non-GP-compatible models
                model.localsolve(**solvekwargs)
            if hasattr(model, "solution"):
                sol = model.solution
            else:
                sol = model.result
            for fn in fns_of_sol:
                fn(sol)
        except RuntimeWarning, e:
            print "RuntimeWarning:", str(e).split("\n")[0]
            print "\n> Running model.debug()"
            model.debug()

    resolve()

    return widgets.interactive(resolve, **ranges_out)


# pylint: disable=too-many-locals
def modelcontrolpanel(model, showvars=(), fns_of_sol=None, **solvekwargs):
    """Easy model control in IPython / Jupyter

    Like interact(), but with the ability to control sliders and their showvars
    live. args and kwargs are passed on to interact()
    """

    freevars = set(model.varkeys).difference(model.substitutions)
    freev_in_showvars = False
    for var in showvars:
        if var in model.varkeys and freevars.intersection(model.varkeys[var]):
            freev_in_showvars = True
            break

    if fns_of_sol is None:
        def __defaultfn(solution):
            "Display function to run when a slider is moved."
            # NOTE: if there are some freevariables in showvars, filter
            #       the table to show only those and the slider constants
            print solution.summary(showvars if freev_in_showvars else ())

        __defaultfntable = __defaultfn

        fns_of_sol = [__defaultfntable]

    sliders = model.interact(fns_of_sol, showvars, **solvekwargs)
    sliderboxes = []
    for sl in sliders.children:
        cb = widgets.Checkbox(value=True, width="3ex")
        unit_latex = sl.varkey.latex_unitstr()
        if unit_latex:
            unit_latex = r"$"+unit_latex+"$"
        units = widgets.Label(value=unit_latex)
        units.font_size = "1.16em"
        box = widgets.HBox(children=[cb, sl, units])
        link((box, 'visible'), (cb, 'value'))
        sliderboxes.append(box)

    widgets_css = widgets.HTML("""<style>
    [style="font-size: 1.16em;"] { padding-top: 0.25em; }
    [style="width: 3ex; font-size: 1.165em;"] { padding-top: 0.2em; }
    .widget-numeric-text { width: auto; }
    .widget-numeric-text .widget-label { width: 20ex; }
    .widget-numeric-text .form-control { background: #fbfbfb; width: 8.5ex; }
    .widget-slider .widget-label { width: 20ex; }
    .widget-checkbox .widget-label { width: 15ex; }
    .form-control { border: none; box-shadow: none; }
    </style>""")
    settings = [widgets_css]
    for sliderbox in sliderboxes:
        settings.append(create_settings(sliderbox))
    sweep = widgets.Checkbox(value=False, width="3ex")
    label = ("Plot top sliders against: (separate with two spaces)")
    boxlabel = widgets.Label(value=label, width="200ex")
    y_axes = widgets.Text(value="none", width="20ex")

    def append_plotfn():
        "Creates and adds plotfn to fn_of_sols"
        from . import plot_1dsweepgrid
        yvars = [model.cost]
        for varname in y_axes.value.split("  "):  # pylint: disable=no-member
            varname = varname.strip()
            try:
                yvars.append(model[varname])
            except:  # pylint: disable=bare-except
                break
        ranges = {}
        for sb in sliderboxes[1:]:
            if sb.visible and len(ranges) < 3:
                slider = sb.children[1]
                ranges[slider.varkey] = (slider.min, slider.max)

        def __defaultfn(sol):
            "Plots a 1D sweep grid, starting from a single solution"
            fig, _ = plot_1dsweepgrid(model, ranges, yvars, origsol=sol,
                                      verbosity=0, solver="mosek_cli")
            fig.show()
        fns_of_sol.append(__defaultfn)

    def redo_plots(_):
        "Refreshes the plotfn"
        if fns_of_sol and fns_of_sol[-1].__name__ == "__defaultfn":
            fns_of_sol.pop()  # get rid of the old one!
        if sweep.value:
            append_plotfn()
        if not fns_of_sol:
            fns_of_sol.append(__defaultfntable)
        sl.value = sl.value*(1.000001)  # pylint: disable=undefined-loop-variable

    sweep.observe(redo_plots, "value")
    y_axes.on_submit(redo_plots)
    sliderboxes.insert(0, widgets.HBox(children=[sweep, boxlabel, y_axes]))
    tabs = widgets.Tab(children=[widgets.VBox(children=sliderboxes,
                                              padding="1.25ex")])

    tabs.set_title(0, 'Sliders')

    return tabs


def create_settings(box):
    "Creates a widget Container for settings and info of a particular slider."
    _, slider, sl_units = box.children

    enable = widgets.Checkbox(value=box.visible, width="3ex")
    link((box, 'visible'), (enable, 'value'))

    def slider_link(obj, attr):
        "Function to link one object to an attr of the slider."
        # pylint: disable=unused-argument
        def link_fn(name, new_value):
            "How to update the object's value given min/max on the slider. "
            if new_value >= slider.max:
                slider.max = new_value
            # if any value is greater than the max, the max slides up
            # however, this is not held true for the minimum, because
            # during typing the max or value will grow, and we don't want
            # to permanently anchor the minimum to unfinished typing
            if attr is "max" and new_value <= slider.value:
                if slider.max >= slider.min:
                    slider.value = new_value
                else:
                    pass  # bounds nonsensical, probably because we picked up
                          # a small value during user typing.
            elif attr is "min" and new_value >= slider.value:
                slider.value = new_value
            setattr(slider, attr, new_value)
            slider.step = (slider.max - slider.min)/24.0
        obj.on_trait_change(link_fn, "value")
        link((slider, attr), (obj, "value"))

    text_html = "<span class='form-control' style='width: auto;'>"
    setvalue = widgets.FloatText(value=slider.value,
                                 description=slider.description)
    slider_link(setvalue, "value")
    fromlabel = widgets.HTML(text_html + "from")
    setmin = widgets.FloatText(value=slider.min, width="10ex")
    slider_link(setmin, "min")
    tolabel = widgets.HTML(text_html + "to")
    setmax = widgets.FloatText(value=slider.max, width="10ex")
    slider_link(setmax, "max")

    units = widgets.Label()
    units.width = "6ex"
    units.font_size = "1.165em"
    link((sl_units, 'value'), (units, 'value'))
    descr = widgets.HTML(text_html + slider.varkey.descr.get("label", ""))
    descr.width = "40ex"

    return widgets.HBox(children=[enable, setvalue, units, descr,
                                  fromlabel, setmin, tolabel, setmax],
                        width="105ex")
