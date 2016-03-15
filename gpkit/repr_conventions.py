"Repository for representation standards"


def _repr(self):
    "Returns namespaced string."
    return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))


def _str(self):
    "Returns default string."
    return self.str_without()


def _repr_latex_(self):
    "Returns default latex for automatic iPython Notebook rendering."
    return "$$"+self.latex()+"$$"
