def chooser(cost, constraints, solver, options):
  if solver == 'cvxopt':
    return cvxopt_solver([cost]+constraints, options)
  else:
    raise Exception, "That solver is not implemented!"

def cvxopt_solver(posynomials, options):
  from cvxopt import matrix, log, exp, solvers
  from itertools import chain
  
  solvers.options.update(options)

  freevars = set().union(*[p.vars for p in posynomials])
  monomials = list(chain(*[p.monomials for p in posynomials]))

  ## CVXopt stuff ##
  # K: number of monomials (columns of F) present in each constraint
  K = [len(p.monomials) for p in posynomials]
  # g: constant coefficients of the various monomials in F
  g = log(matrix([m.c for m in monomials]))
  # F: exponents of the various free variables for each monomial
  F = matrix([[float(m.exps.get(v, 0)) 
                    for m in monomials]
                      for v in freevars])
  # For more details on the format of these matrixes:
  #  http://cvxopt.org/userguide/solvers.html?highlight=gp#cvxopt.solvers.gp

  solution = solvers.gp(K, F, g)
  # TODO: catch errors, delays, etc.
  return dict(zip(freevars, exp(solution['x'])))
