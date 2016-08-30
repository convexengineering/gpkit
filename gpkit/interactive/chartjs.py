"""Interacive tools that use http://www.chartjs.org/"""
from string import Template
from itertools import count
from IPython.display import HTML, display, Javascript
from gpkit.small_scripts import mag


CHARTJSSRC = ("<script src='https://cdnjs.cloudflare.com/ajax/libs/"
              "Chart.js/1.0.2/Chart.min.js' />")


class BarChart(object):
    "See init"

    new_unnamed_id = count().next

    def __init__(self, varnames):  # ,options=None
        """A Python object representing a Chart.js chart
        Can be interacted with as a GPkit widget in Ipython notebook
        """
        self.idnum = self.__class__.new_unnamed_id()
        self.name = "GPkit_%s_%s" % (self.__class__.__name__, self.idnum)
        self.varnames = varnames
        display(HTML(CHARTJSSRC))
        display(self.get_canvas())
        self.updates = 0

    def get_canvas(self):
        "Get the HTML for displaying the chart canvas"
        cstr = "<canvas id='%s' width='%s' height='%s'></canvas>"
        return HTML(cstr % (self.name, 300, 300))

    def update(self, solution):
        "Update the chart based upon a new solution"
        valuedict = solution["variables"]
        if self.updates == 0:  # first update
            self.create_jsobj(valuedict)
        else:
            updates = ""
            for i, varname in enumerate(self.varnames):
                updates += ("%s.datasets[0].bars[%i].value = %f \n"
                            % (self.name, i, mag(valuedict[varname])))
            display(Javascript(updates + "%s.update()" % self.name))
        self.updates += 1

    def create_jsobj(self, data):
        "Create and display the javascript object for this chart"
        labels = ", ".join(['"%s"' % vn for vn in self.varnames])
        datarray = ", ".join([str(mag(data[varname]))
                              for varname in self.varnames])
        js_init = Template("""
        var data = {
            labels: [$labels],
            datasets: [
                {
                    data: [$datarray],
                    fillColor: "rgba(151,187,205,0.5)",
                    strokeColor: "rgba(151,187,205,0.8)",
                    highlightFill: "rgba(151,187,205,0.75)",
                    highlightStroke: "rgba(151,187,205,1)",
                }]}
        var ctx = document.getElementById("$name").getContext("2d");
        window.$name = new Chart(ctx).Bar(data,
                                          {animationSteps: 1}
                                          );
        """).substitute(name=self.name, labels=labels, datarray=datarray)
        display(Javascript(js_init))
