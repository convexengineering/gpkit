from IPython.html.widgets import interactive
from ..small_scripts import unitstr
from ..geometric_program import GPSolutionArray
from IPython.display import Math, display, HTML
from string import Template
import itertools
import numpy as np


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

new_jswidget_id = itertools.count().next

def jswidget(gp, ractorfn, after, ranges):
    widget_id = "jswidget_"+str(new_jswidget_id())
    display(HTML("<script id='%s-after' type='text/throwaway'>%s</script>" % (widget_id, after)))
    display(HTML("<script>var %s = {storage: [], n:%i, ranges: {}, after: document.getElementById('%s-after').innerText, bases: [1] }</script>" % (widget_id, len(ranges), widget_id)))

    container_id = widget_id + "_container"
    display(HTML("<div id='%s'></div><style>#%s td {text-align: right; border: none !important;}\n#%s tr {border: none !important;}\n#%s table {border: none !important;}\n</style>" % (container_id, container_id, container_id, container_id)))

    template_id = widget_id + "_template"
    template = '<script id="%s" type="text/ractive"><table>' % template_id
    ctrl_template = Template('<tr><td>$var</td><input value="{{$varname}}" type="range" min="0" max="$len" step="1"><td><span id="$w-$varname"></span></td></tr>\n')
    data_init = ""

    subs = {}
    lengths = []
    bases = []

    varkeys = gp.unsubbed.var_locs.keys()

    for var, values in ranges.items():
        mini, maxi, step = values
        length = int((maxi-mini)/step) + 1
        lengths.append(length)
        bases.append(np.prod(lengths))
        array = map(lambda x: mini + x*step, range(length))
        if var in varkeys:
            subs[varkeys[varkeys.index(var)]] = ("sweep", array)

    # bug involves things resizing mysteriously when there's >4 vars
    # kinda like swapping wall & floor, but not quite...

    i = 0
    for var, sweepval in subs.items():
        array = sweepval[1]
        varname = "var" + str(i)
        display(HTML("<script>%s.ranges.%s = %s\n%s.bases.push(%i)</script>" % (widget_id, varname, array, widget_id, bases[i])))
        template += ctrl_template.substitute(w=widget_id,
                                             var=("$%s$" % var),
                                             varname=varname,
                                             len=len(array)-1)
        data_init += "%s: %i, " % (varname, (len(array)-1)/2)
        i += 1

    evalarray = [""]*(np.prod(lengths))

    gp.sweep = {}
    gp.prewidget = gp.last
    gp.sub(subs, replace=True)
    sol = gp.solve(printing=False, skipfailures=True)
    for j in range(len(sol)):
        solj = sol.atindex(j)
        soljv = solj["variables"]
        idxs = [subs[var][1].index(soljv[var]) for var in subs]
        k = sum(np.array(idxs) * np.array([1]+bases[:-1]))
        evalarray[k] = ractorfn(GPSolutionArray(solj))
    display(HTML("<script> %s.storage = %s </script>" % (widget_id, evalarray)))
    gp.load(gp.prewidget, printing=False)

    display(HTML(template + "</table></script>"))

    loader = Template("""getScript('http://cdn.ractivejs.org/latest/ractive.min.js', function() {
          $w.ractive = new Ractive({
          el: '$container_id',
          template: '#$template_id',
          magic: true,
          data: {$data_init},
          onchange: function() {
              var idxsum = 0
              for (var i=0; i<$w.n; i++) {
                  varname = 'var'+i
                  idx = $w.ractive.data[varname]
                  document.getElementById("$w-"+varname).innerText = Math.round(100*$w.ranges[varname][idx])/100
                  idxsum += idx*$w.bases[i]
              }
              if ($w.storage[idxsum] === "") {
                r.infeasibilitywarning = "Infeasible problem"
              } else {
                r.infeasibilitywarning = ""
                eval($w.storage[idxsum] + $w.after)
              }
            }
        });

        MathJax.Hub.Typeset()
        $w.ractive.onchange()
})</script>""")

    display(HTML("<script>$."+loader.substitute(w=widget_id,
                                   container_id=container_id,
                                   template_id=template_id,
                                   data_init=data_init)))


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
