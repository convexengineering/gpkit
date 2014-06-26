def chooser(cost, constraints, solver, options):
  if solver == 'cvxopt':
    return cvxopt([cost]+constraints, options)
  else:
    raise Exception, "That solver is not implemented!"

def cvxopt(posynomials, options):
  # the first posynomial will be the cost function

  from cvxopt import matrix, log, exp, solvers
  from itertools import chain
  solvers.options.update(options)

  freevars = set().union(*[p.vars for p in posynomials])
  monomials = list(chain(*[p.monomials for p in posynomials]))

  #  See http://cvxopt.org/userguide/solvers.html?highlight=gp#cvxopt.solvers.gp
  #    for more details on the format of these matrixes

  # K: number of monomials (columns of F) present in each constraint
  K = [len(p.monomials) for p in posynomials]
  # g: constant coefficients of the various monomials in F
  g = log(matrix([m.c for m in monomials]))
  # F: exponents of the various control variables for each of the needed monomials
  F = matrix([[float(m.exps.get(v, 0)) for m in monomials] for v in freevars])

  return dict(zip(freevars, exp(solvers.gp(K, F, g)['x'])))
  