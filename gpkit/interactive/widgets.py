"Interactive GPkit widgets for iPython notebook"
import ipywidgets as widgets
from traitlets import link
from ..small_scripts import is_sweepvar
from ..small_classes import Numbers
from ..exceptions import InvalidGPConstraint


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
        for k, v in model.substitutions.items():
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
                varkey_latex = "$"+k.latex(excluded=["models"])+"$"
                floatslider = widgets.FloatSlider(min=vmin, max=vmax,
                                                  step=vstep, value=v,
                                                  description=varkey_latex)
                # TODO: find way to set width across ipython versions
                # floatslider.width = "20ex"
                floatslider.varkey = k
                ranges[str(k)] = floatslider

    if fn_of_sol is None:
        # pylint: disable=function-redefined
        def fn_of_sol(solution):
            "Display function to run when a slider is moved."
            if hasattr(solution, "table"):
                tables = ["cost", "freevariables", "sensitivities"]
                print solution.table(tables)
            else:
                print solution["variables"]

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
            fn_of_sol(sol)
        except RuntimeWarning:
            raise
            # TODO: implement some nicer feasibility warning, like the below
            # out = "THE PROBLEM IS INFEASIBLE"
            # try:
            #     const_feas = model.feasibility(["constants"])
            #     out += "\n    but would become with this substitution:\n"
            #     out += str(const_feas)
            # except:
            #     pass
            # finally:
            #     print(out)

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
        unit_latex = sl.varkey.latex_unitstr()
        if unit_latex:
            unit_latex = r"$\scriptsize"+unit_latex+"$"
        units = widgets.Latex(value=unit_latex)
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
    # model_latex = "$"+model.latex(show_subs=False)+"$"
    # model_eq = widgets.Latex(model_latex)
    tabs = widgets.Tab(children=[widgets.Box(children=sliderboxes,
                                             padding="1.25ex"),
                                 widgets.Box(children=settings,
                                             padding="1.25ex")])
                                # TODO: fix model equation display
                                # widgets.Box(children=[model_eq],
                                #             padding="1.25ex")])

    tabs.set_title(0, 'Variable Sliders')
    tabs.set_title(1, 'Slider Settings')
    #tabs.set_title(2, 'Model Equations')

    return tabs


def create_settings(box):
    "Creates a widget Container for settings and info of a particular slider."
    _, slider, sl_units = box.children

    enable = widgets.Checkbox(value=box.visible)
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
    setmin = widgets.FloatText(value=slider.min)
    slider_link(setmin, "min")
    tolabel = widgets.HTML(text_html + "to")
    setmax = widgets.FloatText(value=slider.max)
    slider_link(setmax, "max")

    units = widgets.Latex(value="")
    units.width = "6ex"
    units.font_size = "1.165em"
    link((sl_units, 'value'), (units, 'value'))
    descr = widgets.HTML(text_html + slider.varkey.descr.get("label", ""))
    descr.width = "40ex"

    return widgets.HBox(children=[enable, setvalue, units, descr,
                                  fromlabel, setmin, tolabel, setmax])
