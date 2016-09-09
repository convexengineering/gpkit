"Implements LinkedConstraintSet"
from .set import ConstraintSet
from ..varkey import VarKey
from .. import SignomialsEnabled


class LinkedConstraintSet(ConstraintSet):
    """A ConstraintSet that links duplicate variables in its constraints

    VarKeys with the same `.str_without(["models"])` are linked.

    The new linking varkey will have the same attributes as the first linked
    varkey of that name, without any value, models, or modelnums.

    If any of the constraints have a substitution for a linked varkey,
    the linking varkey will have that substitution as well; if more than one
    linked varkey has a substitution a ValueError will be raised.

    Arguments
    ---------
    constraints: iterable
        valid argument to ConstraintSet

    include_only: set
        whitelist of variable names to include

    exclude: set
        blacklist of variable names, supercedes include_only
    """
    def __init__(self, constraints, include_only=None, exclude=None):
        ConstraintSet.__init__(self, constraints)
        exclude = frozenset(exclude) if exclude else frozenset()
        include_only = frozenset(include_only) if include_only else frozenset()
        linkable = set()
        for varkey in self.varkeys:
            name_without_model = varkey.str_without(["models"])
            if include_only and varkey.name not in include_only:
                continue
            if varkey.name in exclude:
                continue
            if len(self.varkeys[name_without_model]) > 1:
                linkable.add(name_without_model)
        self.linked, self.reverselinks = {}, {}
        for name in linkable:
            vks = self.varkeys[name]
            sub, subbed_vk = None, None
            for vk in vks:
                if vk in self.substitutions:
                    if sub is None or sub == self.substitutions[vk]:
                        subbed_vk = vk
                        sub = self.substitutions[vk]
                        del self.substitutions[vk]
                    else:
                        raise ValueError("substitution conflict: could not"
                                         " link because %s was set to %s but"
                                         " %s was set to %s" % (
                                             subbed_vk, sub,
                                             vk, self.substitutions[vk]))
            # vks is a set, so it's convenient to use the loop variable here
            # since we've already verified above that vks is not null
            descr = dict(vk.descr)  # pylint: disable=undefined-loop-variable
            descr.pop("value", None)
            descr.pop("models", None)
            descr.pop("modelnums", None)
            newvk = VarKey(**descr)
            if sub:
                self.substitutions[newvk] = sub
            self.linked.update(dict(zip(vks, len(vks)*[newvk])))
            self.reverselinks[newvk] = vks
        with SignomialsEnabled():  # since we're just substituting varkeys.
            self.subinplace(self.linked)
        for constraint in self:
            if hasattr(constraint, "reset_varkeys"):
                constraint.reset_varkeys()
        self.reset_varkeys()

    def process_result(self, result):
        super(LinkedConstraintSet, self).process_result(result)
        for k in ["constants", "variables", "freevariables", "sensitivities"]:
            resultdict = result[k]
            if k == "sensitivities":  # get ["sensitivities"]["constants"]
                resultdict = resultdict["constants"]
            for newvk, oldvks in self.reverselinks.items():
                if newvk in resultdict:
                    for oldvk in oldvks:
                        units = newvk.units if hasattr(newvk.units, "to") else 1
                        resultdict[oldvk] = resultdict[newvk]*units
