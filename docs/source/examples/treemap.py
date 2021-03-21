"Treemap example"
import plotly  # pylint: disable=unused-import
from gpkit.interactive.plotting import treemap
from performance_modeling import M

fig = treemap(M)
# plotly.offline.plot(fig, filename="treemap.html")  # uncomment to show

fig = treemap(M, itemize="constraints", sizebycount=True)
# plotly.offline.plot(fig, filename="sizedtreemap.html")  # uncomment to show
