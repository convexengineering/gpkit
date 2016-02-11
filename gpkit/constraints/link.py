from . import ConstraintSet
from ..varkey import VarKey


class LinkConstraint(ConstraintSet):
    def __init__(self, constraints, include_only=None, exclude=None):
        ConstraintSet.__init__(self, constraints)
        varkeys = self.varkeys
        linkable = set()
        for varkey in self.varkeys:
            if len(self.varkeys[varkey.name]) > 1:
                linkable.add(varkey.name)
        if include_only:
            linkable &= set(include_only)
        if exclude:
            linkable -= set(exclude)
        self.linked = {}
        for name in linkable:
            vks = self.varkeys[name]
            descr = dict(vks[0].descr)
            value = descr.pop("value", None)
            for vk in vks[1:]:
                value = self.substitutions.pop(vk, value)
            descr.pop("models", None)
            descr.pop("modelnums", None)
            newvk = VarKey(**descr)
            if value:
                self.substitutions[newvk] = value
            self.linked.update(dict(zip(vks, len(vks)*[newvk])))
        self.sub(self.linked)
