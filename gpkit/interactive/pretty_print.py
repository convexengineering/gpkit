try:
    from IPython.display import Latex
except ImportError:
    pass

def pretty_print(obj):
	try:
		return Latex("$$" + obj._latex() + "$$")
	except AttributeError:
		print "Can not pretty-print object without a _latex method."
