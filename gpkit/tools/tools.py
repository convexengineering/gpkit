"""Non-application-specific convenience methods for GPkit"""
import numpy as np
from ..nomials import Variable, VectorVariable
from ..nomials import NomialArray


def te_exp_minus1(posy, nterm):
    """Taylor expansion of e^{posy} - 1

    Arguments
    ---------
    posy : gpkit.Posynomial
        Variable or expression to exponentiate
    nterm : int
        Number of non-constant terms in resulting Taylor expansion

    Returns
    -------
    gpkit.Posynomial
        Taylor expansion of e^{posy} - 1, carried to nterm terms
    """
    res = 0
    factorial_denom = 1
    for i in range(1, nterm + 1):
        factorial_denom *= i
        res += posy**i / factorial_denom
    return res


def te_secant(var, nterm):
    """Taylor expansion of secant(var).

    Arguments
    ---------
    var : gpkit.monomial
      Variable or expression argument
    nterm : int
        Number of non-constant terms in resulting Taylor expansion

    Returns
    -------
    gpkit.Posynomial
        Taylor expansion of secant(x), carried to nterm terms
    """
    # The first 12 Euler Numbers
    E2n = np.asarray([1,
                      5,
                      61,
                      1385,
                      50521,
                      2702765,
                      199360981,
                      19391512145,
                      2404879675441,
                      370371188237525,
                      69348874393137901,
                      15514534163557086905])
    if nterm > 12:
        n_extend = np.asarray(range(13, nterm+1))
        E2n_add = (8 * np.sqrt(n_extend/np.pi)
                   * (4*n_extend/(np.pi * np.exp(1)))**(2*n_extend))
        E2n = np.append(E2n, E2n_add)

    res = 1
    factorial_denom = 1.
    for i in range(1, nterm + 1):
        factorial_denom *= ((2*i)*(2*i-1))
        res += E2n[i-1] / factorial_denom * var**(2*i)
    return res


def te_tangent(var, nterm):
    """Taylor expansion of tangent(var).

    Arguments
    ---------
    var : gpkit.monomial
      Variable or expression argument
    nterm : int
        Number of non-constant terms in resulting Taylor expansion

    Returns
    -------
    gpkit.Posynomial
        Taylor expansion of tangent(x), carried to nterm terms
    """
    if nterm > 15:
        raise NotImplementedError("Tangent expansion not implemented above"
                                  " 15 terms")

    # The first 15 Bernoulli Numbers
    B2n = np.asarray([1/6.,
                      -1/30.,
                      1/42.,
                      -1/30.,
                      5/66.,
                      -691/2730.,
                      7/6.,
                      -3617/510.,
                      43867/798.,
                      -174611/330.,
                      854513/138.,
                      -236364091/2730.,
                      8553103/6.,
                      -23749461029/870.,
                      8615841276005/14322.])

    res = 0
    factorial_denom = 1
    for i in range(1, nterm + 1):
        factorial_denom *= ((2*i)*(2*i-1))
        res += ((-1)**(i-1) * 2**(2*i) * (2**(2*i) - 1) *
                B2n[i-1] / factorial_denom * var**(2*i-1))
    return res


# pylint: disable=too-many-locals
def composite_objective(*objectives, **kwargs):
    "Creates a cost function that sweeps between multiple objectives."
    objectives = list(objectives)
    n = len(objectives)
    k = kwargs.get("k", 4)
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

    sweepvals = np.empty(n-1, dtype="object")
    for i in range(n-1):
        sweepvals[i] = ("sweep", sweeps[i])
    ws = VectorVariable(n-1, "w_{CO}", sweepvals, "-")
    w_s = []
    for w in ws:
        descr = dict(w.descr)
        del descr["value"]
        descr["name"] = "v_{CO}"
        w_s.append(Variable(value=lambda const: 1-const[w.key], **descr))  # pylint: disable=cell-var-from-loop
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
