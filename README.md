# gpkit #

Python tools for defining and manipulating geometric programming models.

Interfaces with either the [MOSEK](http://mosek.com) or [CVXopt](http://cvxopt.org/) solvers.

Installation instructions are below.

## Introduction ##

### What does gpkit look like?

Excerpt from an [aircraft design application](http://nbviewer.ipython.org/github/appliedopt/gpkit/blob/master/examples/ipynb/simpleaircraft.ipynb):

```python
# Note: in Python, '**' serves as the power ('^') operator

gpkit.GP( # minimize                            # What's the lowest
         0.5*rho*S*C_D*V**2,                    # [N] TOTAL DRAG FORCE
         [ # subject to                         # That we can get, with our
          Re <= (rho/mu)*V*(S/A)**0.5,          # flow characteristics,
          C_f >= 0.074/Re**0.2,                 # turbulent BL approximation,
          C_D >= C_D_fuse + C_D_wpar + C_D_ind  # the above 'drag model',
          W <= 0.5*rho*S*C_L*V**2,              # flight at cruising,
          W <= 0.5*rho*S*C_Lmax*V_min**2,       # flight at takeoff,     
          W >= W_0 + W_w,                       # the plane's weight, and
          W_w >= W_w_surf + W_w_strc,           # the above 'wing-weight model'?
         ], solver='mosek')
 ```

### What can a geometric program do?

A geometric program (GP) can solve any optimization problem where [posynomials](http://en.wikipedia.org/wiki/Posynomial) form both the cost function (what you're trying to minimize or maximize, e.g. airplane fuel consumption) and the constraints (equations that have to be true, e.g. that the plane can take off). 

### Why are geometric programs useful?

It turns out that [they have some nice mathematical properties](http://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf):
  - it's easy to check if something is a posynomial
  - they're quick to solve, which is good for large problems and trade-off analysis
  - solving a GP gives you an auomatic sensitivity analysis (via its dual)
  - infeasible GP can be examined to figure out how infeasible they are and which constraints are causing the most trouble

Geometric programs might also have nice social properties:
  - it's clear when you can't turn an equation into a posynomial;
    - this categorizes the design space into variables computers should definitely be choosing, and those that might be harder to solve for
  - many engineering equations are already posynomials
  - posynomial models written by different people are easy to bring together;
      - adding a new model won't bring the whole optimization down

## Installation ##

### Mac

1. Install [Anaconda](continuum.io/downloads) for your platform
   - (Mac) Install developer tools first
   - If you don't want to install Anaconda, the packages useful for running gpkit are numpy, scipy, sympy, and ipython notebook
2. Install ctypesgen by entering `pip install ctypesgen` at a terminal
3. Install [cvxopt](http://cvxopt.org/download/index.html) 
  - (Mac) Just run `python setup.py install` in the downloaded cvxopt folder
4. Install [Mosek](mosek.com/resources/downloads)
  -  (Mac/Linux) Move the downloaded mosek folder to your home directory
5. Get a Mosek [academic license file](license.mosek.com/academic)
  -  (Mac/Linux) Put the license file in ~/mosek/
  - (Windows) Put the license file in the folder Users/(your username)/mosek, creating that folder if necessary
6. Download gpkit
7. Run `python build.py` in the gpkit folder
8. Test your install by running `ipython -c "%run gpkit/tests/run_tests.py"`