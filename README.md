# gpkit #

Python package for defining and manipulating geometric programming models.

Interfaces with multiple solvers -- currently [MOSEK](http://mosek.com) and [CVXopt](http://cvxopt.org/) are supported.

Installation instructions are below.

## Introduction ##

### What does gpkit look like?

Excerpt from an [aircraft design application](http://nbviewer.ipython.org/github/appliedopt/gpkit/blob/master/gpkit/examples/simpleaircraft.ipynb):

```python
# Note: in Python, '**' serves as the power ('^') operator

gpkit.GP( # minimize                            # What's the lowest
         0.5*rho*S*C_D*V**2,                    # [N] TOTAL DRAG FORCE
         [ # subject to                         # That we can get, with our
          Re == (rho/mu)*V*(S/A)**0.5,          # flow characteristics,
          C_f == 0.074/Re**0.2,                 # turbulent BL approximation,
          C_D == C_D_fuse + C_D_wpar + C_D_ind  # the above 'drag model',
          W <= 0.5*rho*S*C_L*V**2,              # flight at cruising,
          W <= 0.5*rho*S*C_Lmax*V_min**2,       # flight at takeoff,     
          W == W_0 + W_w,                       # the plane's weight, and
          W_w == W_w_surf + W_w_strc,           # the above 'wing-weight model'?
         ], solver='mosek')
 ```

### What can a geometric program do?

A geometric program (GP) can solve any optimization problem where [posynomials](http://en.wikipedia.org/wiki/Posynomial) form both the cost function (what you're trying to minimize or maximize, e.g. airplane fuel consumption) and the constraints (relationships that have to be true, e.g. that the plane can take off). 

### Why are geometric programs useful?

It turns out that [they have some nice mathematical properties](http://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf):
  - it's easy to check if something is a posynomial
  - they're quick to solve, which is good for large problems and trade-off analysis
  - solving a GP gives you an automatic sensitivity analysis (via its dual)
  - an infeasible GP can be examined to figure out how infeasible it is and which constraints are causing the most trouble

Geometric programs might also have nice social properties:
  - it's clear when you can't turn an equation into a posynomial;
    - this categorizes the design space into variables computers should definitely be choosing, and those that might be harder to solve for
  - many engineering equations are already posynomials
  - posynomial models written by different people are easy to bring together;
      - adding a new model won't bring the whole optimization down

## Installation ##

1. Install the Python 2.7 version of [Anaconda](http://continuum.io/downloads):
   - (Mac) If `which gcc` does not return anything, install the [Apple Command Line Tools](https://developer.apple.com/downloads/index.action?=command%20line%20tools).
   - If you don't want to install Anaconda, you'll need the python packages numpy and ctypesgen, and might find pip, sympy, and iPython Notebook to be useful as well.
2. Install a solver: (gpkit currently supports both CVXOPT and MOSEK)
  - Download [CVXOPT](http://cvxopt.org/download/index.html):
    - (Mac/Linux) Run `python setup.py install` in the `cvxopt` folder, as noted [here](http://cvxopt.org/install/index.html#standard-installation).
    - (Windows) Follow the steps [here](http://cvxopt.org/install/index.html#building-cvxopt-for-windows).
  - Download [MOSEK](http://mosek.com/resources/downloads):
    -  (Mac OS X) Move the `mosek` folder to your home directory and follow the steps [here](http://docs.mosek.com/7.0/toolsinstall/Mac_OS_X_installation.html).
    -  (Linux) Move the `mosek` folder to your home directory and follow the steps [here](http://docs.mosek.com/7.0/toolsinstall/Linux_UNIX_installation_instructions.html).
    -  (Windows) Follow the steps [here](http://docs.mosek.com/7.0/toolsinstall/Windows_installation.html).
    - Get a MOSEK [academic license file](http://license.mosek.com/academic):
      - (Mac/Linux) Put the license file in `~/mosek/`.
      - (Windows) Put it in `Users/$USERNAME/mosek`, creating that folder if necessary.
3. Run `pip install ctypesgen` and then `pip install https://github.com/appliedopt/gpkit/zipball/master` at a terminal.
  - (Windows) at an "Anaconda Command Prompt".
4. Test your install by running `python -c "import gpkit.tests; gpkit.tests.run()"`.
  - If you haven't installed both MOSEK and CVXOPT, expect a few errors.

If you encounter any bugs during installation, email [eburn@mit.edu](mailto:eburn@mit.edu).
