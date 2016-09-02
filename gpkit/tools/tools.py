"""Non-application-specific convenience methods for GPkit"""
from collections import defaultdict
import numpy as np
from gpkit import Model
from ..nomials import Variable, VectorVariable
from ..nomials import NomialArray
from ..small_scripts import mag


def te_exp_minus1(posy, nterm):
    """Taylor expansion of e^{posy} - 1

    Arguments
    ---------
    posy : gpkit.Posynomial
        Variable or expression to exponentiate
    nterm : int
        Number of terms in resulting Taylor expansion

    Returns
    -------
    gpkit.Posynomial
        Taylor expansion of e^{posy} - 1, carried to nterm terms
    """
    if nterm < 1:
        raise ValueError("Unexpected number of terms, nterm=%s" % nterm)
    res = 0
    factorial_denom = 1
    for i in range(1, nterm + 1):
        factorial_denom *= i
        res += posy**i / factorial_denom
    return res


def composite_objective(*objectives, **kwargs):
    "Creates a cost function that sweeps between multiple objectives."
    objectives = list(objectives)
    n = len(objectives)
    if "k" in kwargs:
        k = kwargs["k"]
    else:
        k = 4
    if "sweep" in kwargs:
        sweeps = [kwargs["sweep"]]*(n-1)
    elif "sweeps" in kwargs:
        sweeps = kwargs["sweeps"]
    else:
        kf = 1/float(k)
        sweeps = [np.linspace(kf, 1-kf, k)]*(n-1)
    if "normsub" in kwargs:
        normalization = [p.sub(kwargs["normsub"]) for p in objectives]
    else:
        normalization = [1]*n

    sweeps = list(zip(["sweep"]*(n-1), sweeps))
    ws = VectorVariable(n-1, "w_{CO}", sweeps, "-")
    w_s = []
    for w in ws:
        descr = dict(w.descr)
        del descr["value"]
        descr["name"] = "v_{CO}"
        w_s.append(Variable(value=('sweep', lambda x: 1-x), args=[w], **descr))
    w_s = normalization[-1]*NomialArray(w_s)*objectives[-1]
    objective = w_s.prod()
    for i, obj in enumerate(objectives[:-1]):
        objective += ws[i]*w_s[:i].prod()*w_s[i+1:].prod()*obj/normalization[i]
    return objective


def mdparse(filename, return_tex=False):
    "Parse markdown file, returning as strings python and (optionally) .tex.md"
    with open(filename) as f:
        py_lines = []
        texmd_lines = []
        block_idx = 0
        in_replaced_block = False
        for line in f:
            line = line[:-1]  # remove newline
            texmd_content = line if not in_replaced_block else ""
            texmd_lines.append(texmd_content)
            py_content = ""
            if line == "```python":
                block_idx = 1
            elif block_idx and line == "```":
                block_idx = 0
                if in_replaced_block:
                    texmd_lines[-1] = ""  # remove the ``` line
                in_replaced_block = False
            elif block_idx:
                py_content = line
                block_idx += 1
                if block_idx == 2:
                    # parsing first line of code block
                    if line[:8] == "#inPDF: ":
                        texmd_lines[-2] = ""  # remove the ```python line
                        texmd_lines[-1] = ""  # remove the #inPDF line
                        in_replaced_block = True
                        if line[8:21] == "replace with ":
                            texmd_lines.append("\\input{%s}" % line[21:])
            elif line:
                py_content = "# " + line
            py_lines.append(py_content)
        if not return_tex:
            return "\n".join(py_lines)
        else:
            return "\n".join(py_lines), "\n".join(texmd_lines)


def mdmake(filename, make_tex=True):
    "Make a python file and (optional) a pandoc-ready .tex.md file"
    mdpy, texmd = mdparse(filename, return_tex=True)
    with open(filename+".py", "w") as f:
        f.write(mdpy)
    if make_tex:
        with open(filename+".tex.md", "w") as f:
            f.write(texmd)
    return open(filename+".py")


def bound_all_variables(model, eps=1e-30, lower=None, upper=None):
    "Returns model with additional constraints bounding all free variables"
    lb = lower if lower else eps
    ub = upper if upper else 1/eps
    constraints = []
    for varkey in model.varkeys:
        units = varkey.descr.get("units", 1)
        constraints.append([ub*units >= Variable(**varkey.descr),
                            Variable(**varkey.descr) >= lb*units])
    m = Model(model.cost, [constraints, model], model.substitutions)
    m.bound_all = {"lb": lb, "ub": ub, "varkeys": model.varkeys}
    return m


# pylint: disable=too-many-locals
def determine_unbounded_variables(model, solver=None, verbosity=0,
                                  eps=1e-30, lower=None, upper=None, **kwargs):
    "Returns labeled dictionary of unbounded variables."
    m = bound_all_variables(model, eps, lower, upper)
    sol = m.solve(solver, verbosity, **kwargs)
    lam = sol["sensitivities"]["la"][1:]
    out = defaultdict(list)
    for i, varkey in enumerate(m.bound_all["varkeys"]):
        lam_gt, lam_lt = lam[2*i], lam[2*i+1]
        if abs(lam_gt) >= 1e-7:  # arbitrary threshold
            out["sensitive to upper bound"].append(varkey)
        if abs(lam_lt) >= 1e-7:  # arbitrary threshold
            out["sensitive to lower bound"].append(varkey)
        value = mag(sol["variables"][varkey])
        distance_below = np.log(value/m.bound_all["lb"])
        distance_above = np.log(m.bound_all["ub"]/value)
        if distance_below <= 3:  # arbitrary threshold
            out["value near lower bound"].append(varkey)
        elif distance_above <= 3:  # arbitrary threshold
            out["value near upper bound"].append(varkey)
    return out
