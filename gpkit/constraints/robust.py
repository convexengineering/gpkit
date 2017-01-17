from numpy import sqrt
from scipy.special import erfcinv
from .set import ConstraintSet
from ..import NamedVariables
from ..nomials import Variable

SUPPORTED_DISTRIBUTIONS = ["normal", "lognormal"]
DEFAULT_P_VIOL = 0.05


class Robust(ConstraintSet):
    def __init__(self, robustvarkeys, p_viol):
        ConstraintSet.__init__(self, [])
        medianvars, sigmavars = set(), set()
        self.distr = None
        # sanity check varkeys
        for vk in robustvarkeys:
            if self.distr is None:
                if vk.distr not in SUPPORTED_DISTRIBUTIONS:
                    raise ValueError("Unsupported uncertainty distribution for"
                                     " '%s': '%s'." % (vk, vk.distr))
                self.distr = vk.distr
            elif self.distr != vk.distr:
                raise ValueError("All uncertainty variables must have the"
                                 " same type of distribution.")
            if vk.better is None or abs(vk.better) is not 1:
                raise ValueError("Uncertain variable '%s' must specify a 'bett"
                                 "er' of 1 (larger is better) or -1 (smaller"
                                 " is better) not '%s'." % (vk, vk.better))
            if vk.distr is "normal" and vk.better is not 1:
                raise ValueError("Normally-distributed uncertain variables"
                                 " like '%s' must specify a 'better' of +1"
                                 " not '%s'." % (vk, vk.better))
            if vk.sigma is None or vk.sigma < 0:
                raise ValueError("The sigma of uncertain variable '%s' must"
                                 " be greater than or equal to zero." % vk)
            # make medianvars
            mediandescr = dict(vk.descr)
            for key in ["median", "sigma", "distr"]:
                del mediandescr[key]
            mediandescr["name"] += "_{median}"
            medianvar = Variable(value=vk.median, **mediandescr)
            vk.descr["median"] = medianvar.key
            medianvars.add(medianvar)
            # make sigmavars
            sigmadescr = dict(vk.descr)
            for key in ["median", "sigma", "distr"]:
                del sigmadescr[key]
            sigmadescr["name"] = "\\sigma_{%s}" % sigmadescr["name"]
            sigmavar = Variable(value=vk.sigma, **sigmadescr).key
            vk.descr["sigma"] = sigmavar.key
            sigmavars.add(sigmavar)
        with NamedVariables("Robust"):
            p_viol = Variable("p_{viol}", p_viol, "-",
                              "Probability of constraint violation")
            sigma = Variable("\\Sigma", lambda c: erfcinv(c[p_viol])*sqrt(2),
                             "-", "Radius of uncertainty sphere")
            probvars = set([p_viol, sigma])
        self.unique_varkeys = set(robustvarkeys)
        for var_list in [probvars, medianvars, sigmavars]:
            self.unique_varkeys.update(v.key for v in var_list)
        self.reset_varkeys()
        self.robustvarkeys = robustvarkeys
        for var_list in [probvars, medianvars, sigmavars]:
            self.substitutions.update({v: v.key.value for v in var_list})

    def get_Sigma(self):
        value = self.substitutions["\\Sigma"]
        if hasattr(value, "__call__"):
            return value(gp.robust.substitutions)
        return value

    @classmethod
    def from_model(cls, model, verbosity=1):
        robustvarkeys = set(vk for vk in model.varkeys if vk.median)
        if not robustvarkeys:
            return

        # determine initial p_viol
        if hasattr(model, "p_viol"):
            p_viol = model.p_viol
        else:
            p_viols = [cs.p_viol for cs in model.flat(constraintsets=True)
                       if hasattr(cs, "p_viol")]
            p_viol = min(p_viols) if p_viols else DEFAULT_P_VIOL

        robust = cls(robustvarkeys, p_viol)

        if verbosity > 0:
            for distr in SUPPORTED_DISTRIBUTIONS:
                dvks = [vk for vk in robustvarkeys if vk.distr is distr]
                if dvks:
                    print("%i %sly distributed variables" % (len(dvks), distr)
                          + ": %s" % dvks if len(dvks) < 5 else "")
            print("Maximum probability of constraint violation:"
                  " %i%%" % (100*robust["p_{viol}"].value))

        return robust
