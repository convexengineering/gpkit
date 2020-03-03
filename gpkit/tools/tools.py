"""Non-application-specific convenience methods for GPkit"""
import numpy as np


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
    E2n = np.asarray([1.0,
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
    factorial_denom = 1
    for i in range(1, nterm + 1):
        factorial_denom *= ((2*i)*(2*i-1))
        res = res + var**(2*i) * E2n[i-1] / factorial_denom
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
    B2n = np.asarray([1/6,
                      -1/30,
                      1/42,
                      -1/30,
                      5/66,
                      -691/2730,
                      7/6,
                      -3617/510,
                      43867/798,
                      -174611/330,
                      854513/138,
                      -236364091/2730,
                      8553103/6,
                      -23749461029/870,
                      8615841276005/14322])

    res = 0
    factorial_denom = 1
    for i in range(1, nterm + 1):
        factorial_denom *= ((2*i)*(2*i-1))
        res += ((-1)**(i-1) * 2**(2*i) * (2**(2*i) - 1) *
                B2n[i-1] / factorial_denom * var**(2*i-1))
    return res
