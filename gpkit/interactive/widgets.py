import ipywidgets as widgets
from traitlets import link
from ..small_scripts import is_sweepvar
from ..small_classes import Numbers


def modelinteract(model, ranges=None, fn_of_sol=None, **solvekwargs):
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
    if ranges is None:
        ranges = {}
        for k, v in model.allsubs.items():
            if is_sweepvar(v) or isinstance(v, Numbers):
                if is_sweepvar(v):
                    # By default, sweep variables become sliders, so we
                    # need to to update the model's substitutions such that
                    # the first solve (with no substitutions) reflects
                    # the substitutions created on slider movement
                    sweep = v[1]
                    v = sweep[0]
                    model.substitutions.update({k: v})
                vmin, vmax = v/2.0, v*2.0
                if is_sweepvar(v):
                    vmin = min(vmin, min(sweep))
                    vmax = max(vmax, min(sweep))
                vstep = (vmax-vmin)/24.0
                varkey_latex = "$"+k.latex()+"$"
                floatslider = widgets.FloatSlider(min=vmin, max=vmax,
                                                  step=vstep, value=v,
                                                  description=varkey_latex)
                floatslider.width = "20ex"
                floatslider.varkey = k
                ranges[k._cmpstr] = floatslider

    if fn_of_sol is None:
        def fn_of_sol(solution):
            tables = ["cost", "freevariables", "sweepvariables"]
            if len(solution["freevariables"]) < 20:
                tables.append("sensitivities")
            print solution.table(tables)

    solvekwargs["verbosity"] = 0

    def resolve(**subs):
        model.substitutions.update(subs)
        try:
            try:
                model.solve(**solvekwargs)
            except ValueError:
                model.localsolve(**solvekwargs)
            fn_of_sol(model.solution)
        except RuntimeWarning:
            out = "THE PROBLEM IS INFEASIBLE"
            try:
                const_feas = model.feasibility(["constants"])
                out += "\n    but would become with this substitution:\n"
                out += str(const_feas)
            except:
                pass
            finally:
                print(out)

    resolve()

    return widgets.interactive(resolve, **ranges)


def modelcontrolpanel(model, *args, **kwargs):
    """Easy model control in IPython / Jupyter

    Like interact(), but with the ability to control sliders and their ranges
    live. args and kwargs are passed on to interact()
    """

    sliders = model.interact(*args, **kwargs)
    sliderboxes = []
    for sl in sliders.children:
        cb = widgets.Checkbox(value=True)
        unit_latex = sl.varkey.unitstr
        if unit_latex:
            unit_latex = "$\scriptsize"+unit_latex+"$"
        units = widgets.Latex(value=unit_latex)
        units.font_size = "1.15em"
        box = widgets.HBox(children=[cb, sl, units])
        link((box, 'visible'), (cb, 'value'))
        sliderboxes.append(box)

    settings = []
    for sliderbox in sliderboxes:
        settings.append(create_settings(sliderbox))

    model_latex = "$"+model.latex(show_subs=False)+"$"
    widgets_css = widgets.HTML("""<style>
    [style="font-size: 1.15em;"] { padding-top: 0.25em; }
    .widget-numeric-text { width: auto; }
    .widget-numeric-text .widget-label { width: 15ex; }
    .widget-numeric-text .form-control { background: #fbfbfb; width: 10ex; }
    .widget-slider .widget-label { width: 15ex; }
    .widget-checkbox .widget-label { width: 15ex; }
    .form-control { border: none; box-shadow: none; }
    </style>""")
    model_eq = widgets.Latex(model_latex)
    tabs = widgets.Tab(children=[widgets.Box(children=sliderboxes,
                                             padding="1.25ex"),
                                 widgets.Box(children=settings,
                                             padding="1.25ex"),
                                 widgets.Box(children=[widgets_css, model_eq],
                                             padding="1.25ex")])

    tabs.set_title(0, 'Variable Sliders')
    tabs.set_title(1, 'Slider Settings')
    tabs.set_title(2, 'Model Equations')

    return tabs


def create_settings(box):
    "Creates a widget Container for settings and info of  a particular slider."
    sl_enable, slider, sl_units = box.children

    enable = widgets.Checkbox(value=box.visible)
    link((box, 'visible'), (enable, 'value'))
    value = widgets.FloatText(value=slider.value,
                              description=slider.description)
    link((slider, 'value'), (value, 'value'))
    units = widgets.Latex(value="")
    link((sl_units, 'value'), (units, 'value'))
    units.font_size = "1.15em"
    fromlabel = widgets.HTML("<span class='form-control' style='width: auto;'>"
                             "from")
    setmin = widgets.FloatText(value=slider.min)
    link((slider, 'min'), (setmin, 'value'))
    tolabel = widgets.HTML("<span class='form-control' style='width: auto;'>"
                           "to")
    setmax = widgets.FloatText(value=slider.max)
    link((slider, 'max'), (setmax, 'value'))
    descr = widgets.HTML("<span class='form-control' style='width: auto;'>"
                         + slider.varkey.descr.get("label", ""))
    descr.width = "40ex"

    return widgets.HBox(children=[enable, value, descr,
                                  fromlabel, setmin, tolabel, setmax, units])
