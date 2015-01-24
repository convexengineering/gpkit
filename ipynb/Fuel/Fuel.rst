.. figure::  fuellogo.svg
   :align:   left
   :width: 150 px

AIRPLANE FUEL
=============

*Minimize fuel needed for a plane that can sprint and land quickly.*

Set up the modelling environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First we'll to import GPkit and turn on :math:`\LaTeX` printing for
GPkit variables and equations.

.. code:: python

    import numpy as np
    import gpkit
    import gpkit.interactive
    gpkit.interactive.init_printing()
declare constants
~~~~~~~~~~~~~~~~~

.. code:: python

    mon = gpkit.Variable
    vec = gpkit.VectorVariable
    
    N_lift         = mon("N_{lift}", 6.0, "-", "Wing loading multiplier")
    pi             = mon("\\pi", np.pi, "-", "Half of the circle constant")
    sigma_max      = mon("\\sigma_{max}", 250e6, "Pa", "Allowable stress, 6061-T6")
    sigma_maxshear = mon("\\sigma_{max,shear}", 167e6, "Pa", "Allowable shear stress")
    g              = mon("g", 9.8, "m/s^2", "Gravitational constant")
    w              = mon("w", 0.5, "-", "Wing-box width/chord")
    r_h            = mon("r_h", 0.75, "-", "Wing strut taper parameter")
    f_wadd         = mon("f_{wadd}", 2, "-", "Wing added weight fraction")
    W_fixed        = mon("W_{fixed}", 14.7e3, "N", "Fixed weight")
    C_Lmax         = mon("C_{L,max}", 1.5, "-", "Maximum C_L, flaps down")
    rho            = mon("\\rho", 0.91, "kg/m^3", "Air density, 3000m")
    rho_sl         = mon("\\rho_{sl}", 1.23, "kg/m^3", "Air density, sea level")
    rho_alum       = mon("\\rho_{alum}", 2700, "kg/m^3", "Density of aluminum")
    mu             = mon("\\mu", 1.69e-5, "kg/m/s", "Dynamic viscosity, 3000m")
    e              = mon("e", 0.95, "-", "Wing spanwise efficiency")
    A_prop         = mon("A_{prop}", 0.785, "m^2", "Propeller disk area")
    eta_eng        = mon("\\eta_{eng}", 0.35, "-", "Engine efficiency")
    eta_v          = mon("\\eta_v", 0.85, "-", "Propeller viscous efficiency")
    h_fuel         = mon("h_{fuel}", 42e6, "J/kg", "fuel heating value")
    V_sprint_reqt  = mon("V_{sprintreqt}", 150, "m/s", "sprint speed requirement")
    W_pay          = mon("W_{pay}", 500*9.81, "N")
    R_min          = mon("R_{min}", 1e6, "m", "Minimum airplane range")
    V_stallmax     = mon("V_{stall,max}", 40, "m/s", "Stall speed")
    # sweep variables
    R_min          = mon("R_{min}", np.linspace(1e6, 1e7, 10), "m", "Minimum airplane range")
    V_stallmax     = mon("V_{stall,max}", np.linspace(30, 50, 10), "m/s", "Stall speed")
declare free variables
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    V        = vec(3, "V", "m/s", "Flight speed")
    C_L      = vec(3, "C_L", "-", "Wing lift coefficent")
    C_D      = vec(3, "C_D", "-", "Wing drag coefficent")
    C_Dfuse  = vec(3, "C_{D_{fuse}}", "-", "Fuselage drag coefficent")
    C_Dp     = vec(3, "C_{D_p}", "-", "drag model parameter")
    C_Di     = vec(3, "C_{D_i}", "-", "drag model parameter")
    T        = vec(3, "T", "N", "Thrust force")
    Re       = vec(3, "Re", "-", "Reynold's number")
    W        = vec(3, "W", "N", "Aircraft weight")
    eta_i    = vec(3, "\\eta_i", "-", "Aircraft efficiency")
    eta_prop = vec(3, "\\eta_{prop}", "-")
    eta_0    = vec(3, "\\eta_0", "-")
    W_fuel   = vec(2, "W_{fuel}", "N", "Fuel weight")
    z_bre    = vec(2, "z_{bre}", "-")
    S        = mon("S", "m^2", "Wing area")
    R        = mon("R", "m", "Airplane range")
    A        = mon("A", "-", "Aspect Ratio")
    I_cap    = mon("I_{cap}", "m^4", "Spar cap area moment of inertia per unit chord")
    M_rbar   = mon("\\bar{M}_r", "-")
    P_max    = mon("P_{max}", "W")
    V_stall  = mon("V_{stall}", "m/s")
    nu       = mon("\\nu", "-")
    p        = mon("p", "-")
    q        = mon("q", "-")
    tau      = mon("\\tau", "-")
    t_cap    = mon("t_{cap}", "-")
    t_web    = mon("t_{web}", "-")
    W_cap    = mon("W_{cap}", "N")
    W_zfw    = mon("W_{zfw}", "N", "Zero fuel weight")
    W_eng    = mon("W_{eng}", "N")
    W_mto    = mon("W_{mto}", "N", "Maximum takeoff weight")
    W_pay    = mon("W_{pay}", "N")
    W_tw     = mon("W_{tw}", "N")
    W_web    = mon("W_{web}", "N")
    W_wing   = mon("W_{wing}", "N")
Let's check that the vector constraints are working:

.. code:: python

    W == 0.5*rho*C_L*S*V**2



.. math::

    \begin{bmatrix}{W}_{0} = 0.5S \rho {V}_{0}^{2} {C_L}_{0} & {W}_{1} = 0.5S \rho {V}_{1}^{2} {C_L}_{1} & {W}_{2} = 0.5S \rho {V}_{2}^{2} {C_L}_{2}\end{bmatrix}



Looks good!

Form the optimization problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the 3-element vector variables, indexs 0, 1, and 2 are the outbound,
return and sprint flights.

.. code:: python

    steady_level_flight = (W == 0.5*rho*C_L*S*V**2,
                           T >= 0.5*rho*C_D*S*V**2,
                           Re == (rho/mu)*V*(S/A)**0.5)
    
    landing_fc = (W_mto <= 0.5*rho_sl*V_stall**2*C_Lmax*S,
                  V_stall <= V_stallmax)
    
    sprint_fc = (P_max >= T[2]*V[2]/eta_0[2],
                 V[2] >= V_sprint_reqt)
    
    drag_model = (C_D >= (0.05/S)*gpkit.units.m**2 +C_Dp + C_L**2/(pi*e*A),
                  1 >= (2.56*C_L**5.88/(Re**1.54*tau**3.32*C_Dp**2.62) +
                       3.8e-9*tau**6.23/(C_L**0.92*Re**1.38*C_Dp**9.57) +
                       2.2e-3*Re**0.14*tau**0.033/(C_L**0.01*C_Dp**0.73) +
                       1.19e4*C_L**9.78*tau**1.76/(Re*C_Dp**0.91) +
                       6.14e-6*C_L**6.53/(Re**0.99*tau**0.52*C_Dp**5.19)))
    
    propulsive_efficiency = (eta_0 <= eta_eng*eta_prop,
                             eta_prop <= eta_i*eta_v,
                             4*eta_i + T*eta_i**2/(0.5*rho*V**2*A_prop) <= 4)
    
    # 4th order taylor approximation for e^x
    z_bre_sum = 0
    for i in range(1,5):
        z_bre_sum += z_bre**i/np.math.factorial(i)
    
    range_constraints = (R >= R_min,
                         z_bre >= g*R*T[:2]/(h_fuel*eta_0[:2]*W[:2]),
                         W_fuel/W[:2] >= z_bre_sum)
    
    weight_relations = (W_pay >= 500*g*gpkit.units.kg,
                        W_tw >= W_fixed + W_pay + W_eng,
                        W_zfw >= W_tw + W_wing,
                        W_eng >= 0.0372*P_max**0.8083 * gpkit.units.parse_expression('N/W^0.8083'),
                        W_wing/f_wadd >= W_cap + W_web,
                        W[0] >= W_zfw + W_fuel[1],
                        W[1] >= W_zfw,
                        W_mto >= W[0] + W_fuel[0],
                        W[2] == W[0])
    
    wing_structural_model = (2*q >= 1 + p,
                             p >= 1.9,
                             tau <= 0.15,
                             M_rbar >= W_tw*A*p/(24*gpkit.units.N),
                             .92**2/2.*w*tau**2*t_cap >= I_cap * gpkit.units.m**-4 + .92*w*tau*t_cap**2,
                             8 >= N_lift*M_rbar*A*q**2*tau/S/I_cap/sigma_max * gpkit.units.parse_expression('Pa*m**6'),
                             12 >= A*W_tw*N_lift*q**2/tau/S/t_web/sigma_maxshear,
                             nu**3.94 >= .86*p**-2.38 + .14*p**0.56,
                             W_cap >= 8*rho_alum*g*w*t_cap*S**1.5*nu/3/A**.5,
                             W_web >= 8*rho_alum*g*r_h*tau*t_web*S**1.5*nu/3/A**.5
                             )
.. code:: python

    eqns = (weight_relations + range_constraints + propulsive_efficiency
            + drag_model + steady_level_flight + landing_fc + sprint_fc + wing_structural_model)
    
    gp = gpkit.GP(W_fuel.sum(), eqns)
Design a hundred airplanes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    sol = gp.solve()

.. parsed-literal::

    Using solver 'cvxopt'
    Sweeping 2 variables over 100 passes
    Solving took 9.96 seconds


The "local model" is the power-law tangent to the Pareto frontier,
gleaned from sensitivities.

.. code:: python

    sol["local_model"][0].sub(gp.substitutions)



.. math::

    0.006484\frac{R_{min}}{V_{stall,max}^{0.59}}



plot design frontiers
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    %matplotlib inline
    %config InlineBackend.figure_format = 'retina'
    
    plot_frontiers = gpkit.interactive.plot_frontiers
    plot_frontiers(gp, [V[0], V[1], V[2]])
    plot_frontiers(gp, [S, W_zfw, P_max])
    plot_frontiers(gp, ['S{\\rho_{alum}}', 'S{h_{fuel}}', 'S{A_{prop}}'])


.. image:: Fuel_files/Fuel_17_0.png



.. image:: Fuel_files/Fuel_17_1.png



.. image:: Fuel_files/Fuel_17_2.png


Interactive analysis
~~~~~~~~~~~~~~~~~~~~

Let's investigate it with the
`cadtoons <https://github.com/bqpd/cadtoons>`__ library. Running
``cadtoon.py flightconditions.svg`` in this folder creates an
interactive SVG graphic for us.

First, import the functions to display HTML in iPython Notebook, and the
`ractivejs <http://www.ractivejs.org/>`__ library.

.. code:: python

    from IPython.display import HTML, display
    from string import Template
.. code:: python

    ractor = Template("""
    var W_eng = $W_eng,
        lam = $lam
    
    r.shearinner.scalex = 1-$tcap*10
    r.shearinner.scaley = 1-$tweb*100
    r.airfoil.scaley = $tau/0.13
    r.fuse.scalex = $W_fus/24000
    r.wing.scalex = $b/2/14
    r.wing.scaley = $cr*1.21
    """)
    
    def ractorfn(sol):
        return ractor.substitute(lam = 0.5*(sol(p) - 1), 
                                 b = sol((S*A)**0.5), 
                                 cr = sol(2/(1+0.5*(sol(p) - 1))*(S/A)**0.5),
                                 tcap = sol(t_cap/tau),
                                 tweb = sol(t_web/w),
                                 tau = sol(tau),
                                 W_eng = sol(W_eng),
                                 W_fus = sol(W_mto) - sol(W_wing) - sol(W_eng))
    
    constraints = """
    r.engine1.scale = Math.pow(W_eng/3000, 2/3)
    r.engine2.scale = Math.pow(W_eng/3000, 2/3)
    r.engine1.y = 6*lam
    r.engine2.y = 6*lam
    r.wingrect.scaley = 1-lam
    r.wingrect.y = -6 + 5*lam
    r.wingtaper.scaley = lam
    r.wingtaper.y = 5*lam
    """
    
    def ractivefn(gp):
        sol = gp.solution
        live = "<script>" + ractorfn(sol) + constraints + "</script>"
        display(HTML(live))
        # if you enable the line below, you can try navigating the sliders by sensitivities
        # print sol.table(["cost", "sensitivities"]) 
.. code:: python

    with open("flightconditions.gpkit", 'r') as file:
        display(HTML(file.read()))
    display(HTML("<style> #ractivecontainer"
                 "{position:absolute; height: 0;"
                 "right: 0; top: -6em;} </style>"))


.. raw:: html

    <div id='ractivecontainer'></div>
    <script id='ractivetemplate' type='text/ractive'>
    <svg width="400" height="400" version="1.1">
    	<g transform="scale(4.00)">
    		<g transform="translate(0,-952.36218)">
    			<g transform="matrix(0.35423407,0,0,0.42994087,-35.858031,688.41834)">
    				<g class="plane" transform="matrix({{plane.scalex * plane.scale}}, 0, 0, {{plane.scaley * plane.scale}}, {{plane.x + 126.01282195 * (1-plane.scalex * plane.scale)}}, {{plane.y + 662.03094 * (1-plane.scaley * plane.scale)}})">
    					<path d="m 232.14732,750.54867 0,-126.8125 c 0,-13.41493 20,-12.7923 20,0 l 0,126.8125 z" style="font-size:medium;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;text-indent:0;text-align:start;text-decoration:none;line-height:normal;letter-spacing:normal;word-spacing:normal;text-transform:none;direction:ltr;block-progression:tb;writing-mode:lr-tb;text-anchor:start;baseline-shift:baseline;color:#000000;fill:#000000;fill-opacity:1;stroke:none;stroke-width:1;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate;font-family:Sans;-inkscape-font-specification:Sans" transform="matrix({{fuse.scalex * fuse.scale}}, 0, 0, {{fuse.scaley * fuse.scale}}, {{fuse.x + 242.14732 * (1-fuse.scalex * fuse.scale)}}, {{fuse.y + 687.14242 * (1-fuse.scaley * fuse.scale)}})" class="fuse"></path>
    					<path d="m 303.49107,650.87477 a 5.0005,3.6595115 0 0 0 -4.90625,3.70488 l 0,14.63658 a 5.0005,3.6595115 0 1 0 10,0 l 0,-14.63658 a 5.0005,3.6595115 0 0 0 -5.09375,-3.70488 z" style="font-size:medium;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;text-indent:0;text-align:start;text-decoration:none;line-height:normal;letter-spacing:normal;word-spacing:normal;text-transform:none;direction:ltr;block-progression:tb;writing-mode:lr-tb;text-anchor:start;baseline-shift:baseline;color:#000000;fill:#000000;fill-opacity:1;stroke:none;stroke-width:1;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate;font-family:Sans;-inkscape-font-specification:Sans" transform="matrix({{engine2.scalex * engine2.scale}}, 0, 0, {{engine2.scaley * engine2.scale}}, {{engine2.x + 303.58482 * (1-engine2.scalex * engine2.scale)}}, {{engine2.y + 660.0455 * (1-engine2.scaley * engine2.scale)}})" class="engine2"></path>
    					<path d="m 182.05357,650.87478 a 5.0005,3.7433229 0 0 0 -4.90625,3.78974 l 0,14.97179 a 5.0005,3.7433229 0 1 0 10,0 l 0,-14.97179 a 5.0005,3.7433229 0 0 0 -5.09375,-3.78974 z" style="font-size:medium;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;text-indent:0;text-align:start;text-decoration:none;line-height:normal;letter-spacing:normal;word-spacing:normal;text-transform:none;direction:ltr;block-progression:tb;writing-mode:lr-tb;text-anchor:start;baseline-shift:baseline;color:#000000;fill:#000000;fill-opacity:1;stroke:none;stroke-width:1;marker:none;visibility:visible;display:inline;overflow:visible;enable-background:accumulate;font-family:Sans;-inkscape-font-specification:Sans" transform="matrix({{engine1.scalex * engine1.scale}}, 0, 0, {{engine1.scaley * engine1.scale}}, {{engine1.x + 182.14732 * (1-engine1.scalex * engine1.scale)}}, {{engine1.y + 660.255545 * (1-engine1.scaley * engine1.scale)}})" class="engine1"></path>
    					<path d="m 239.64285,741.66071 -27.5,7.14286 0,10.71428 60,0 0,-10.71428 -27.5,-7.14286 -5,0 z" style="fill:#ffffff;fill-opacity:1;stroke:#000000;stroke-width:1;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"></path>
    					<path d="m 239.64285,741.66071 -27.5,7.14286 0,10.71428 60,0 0,-10.71428 -27.5,-7.14286 -5,0 z" style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:2;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"></path>
    					<g transform="translate(116.42857,10)">
    						<g class="wing" transform="matrix({{wing.scalex * wing.scale}}, 0, 0, {{wing.scaley * wing.scale}}, {{wing.x + 126.01282195 * (1-wing.scalex * wing.scale)}}, {{wing.y + 662.03094 * (1-wing.scaley * wing.scale)}})">
    							<path style="fill:#000000;fill-opacity:1;stroke:none" d="m 5.6979239,661.98376 0,11.62951 239.9543361,0 0,-11.62951 z" transform="matrix({{wingrect.scalex * wingrect.scale}}, 0, 0, {{wingrect.scaley * wingrect.scale}}, {{wingrect.x + 125.67509195 * (1-wingrect.scalex * wingrect.scale)}}, {{wingrect.y + 667.798515 * (1-wingrect.scaley * wingrect.scale)}})" class="wingrect"></path>
    							<path style="fill:#000000;fill-opacity:1;stroke:none" d="m 116.35246,650.44861 -109.9790744,11.62951 239.9543344,0 -109.97907,-11.62951 z" transform="matrix({{wingtaper.scalex * wingtaper.scale}}, 0, 0, {{wingtaper.scaley * wingtaper.scale}}, {{wingtaper.x + 126.3505528 * (1-wingtaper.scalex * wingtaper.scale)}}, {{wingtaper.y + 656.263365 * (1-wingtaper.scaley * wingtaper.scale)}})" class="wingtaper"></path>
    						</g>
    					</g>
    				</g>
    			</g>
    			<g transform="matrix(0.33791059,0,0,0.33791059,-39.058658,848.18441)">
    				<g class="airfoil" transform="matrix({{airfoil.scalex * airfoil.scale}}, 0, 0, {{airfoil.scaley * airfoil.scale}}, {{airfoil.x + 263.55751 * (1-airfoil.scalex * airfoil.scale)}}, {{airfoil.y + 555.132635 * (1-airfoil.scaley * airfoil.scale)}})">
    					<path d="m 116.59001,559.39951 c 0.16625,-9.0575 20.87875,-15.98625 45.98625,-19.4375 42.72125,-4.05625 87.9375,-5.07375 135.115,0.2375 16.51085,1.7946 34.20674,3.82764 51.79946,6.65278 21.25232,3.41282 42.35407,7.98155 61.03429,14.68222 -28.6475,-2.54375 -65.85625,2.9375 -112.59625,8.76875 -49.39875,4.5325 -90.3825,4.37125 -135.3525,0 -31.0175,-2.06125 -46.1475,-5.23625 -45.98625,-10.90375 z" style="fill:#cccccc;stroke:#646464;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"></path>
    					<path d="m 162.57626,539.96201 0,30.10625 c 44.24875,4.74 89.445,4.74 135.58875,0 l 0,-29.86875 c -45.43375,-4.61 -82.8075,-5.17625 -135.58875,-0.2375 z" style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0.85749996;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none"></path>
    					<path d="m 162.57626,539.96201 0,30.10625 c 44.24874,4.74 89.445,4.74 135.58875,0 l 0,-29.86875 c -45.43376,-4.61 -82.80751,-5.17625 -135.58875,-0.2375 z" style="fill:#ffffff;fill-opacity:1;stroke:#000000;stroke-width:0.85749996;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-opacity:1;stroke-dasharray:none" transform="matrix({{shearinner.scalex * shearinner.scale}}, 0, 0, {{shearinner.scaley * shearinner.scale}}, {{shearinner.x + 230.370635 * (1-shearinner.scalex * shearinner.scale)}}, {{shearinner.y + 555.015135 * (1-shearinner.scaley * shearinner.scale)}})" class="shearinner"></path>
    				</g>
    			</g>
    		</g>
    	</g>
    </svg>
    <div style="text-align: right; font-weight: 700; font-size: 2em;">{{infeasibilitywarning}}</div>
        </script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <script>
        var r = {
    infeasibilitywarning: "",
    airfoil: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    engine1: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    engine2: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    fuse: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    plane: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    shearinner: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    wing: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    wingrect: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
    wingtaper: {scalex: 1, scaley: 1, scale: 1, x:0, y:0},
          }
    $.getScript('http://cdn.ractivejs.org/latest/ractive.min.js', function () {
    var ractive = new Ractive({
              el: 'ractivecontainer',
              template: '#ractivetemplate',
              magic: true,
              data: r
            }) })
    </script>

.. code:: python

    gpkit.interactive.widget(gp, ractivefn,
             {"V_{stall,max}": (20, 50, 1),
              "R_{min}": (1e6, 1e7, 0.5e6)})


.. raw:: html

    <script>
    var W_eng = 3597.16200496,
        lam = 0.450000000325
    
    r.shearinner.scalex = 1-0.0205941604758*10
    r.shearinner.scaley = 1-0.000923106165308*100
    r.airfoil.scaley = 0.149999999986/0.13
    r.fuse.scalex = 27800.4600332/24000
    r.wing.scalex = 24.7675549219/2/14
    r.wing.scaley = 2.00316984856*1.21
    
    r.engine1.scale = Math.pow(W_eng/3000, 2/3)
    r.engine2.scale = Math.pow(W_eng/3000, 2/3)
    r.engine1.y = 6*lam
    r.engine2.y = 6*lam
    r.wingrect.scaley = 1-lam
    r.wingrect.y = -6 + 5*lam
    r.wingtaper.scaley = lam
    r.wingtaper.y = 5*lam
    </script>


.. code:: python

    gpkit.interactive.jswidget(gp, ractorfn,
                               constraints, 
               {"V_{stall,max}": (20, 50, 3),
                "R_{min}": (1e6, 1e7, 1e6)})


.. raw:: html

    <script id='jswidget_0-after' type='text/throwaway'>
    r.engine1.scale = Math.pow(W_eng/3000, 2/3)
    r.engine2.scale = Math.pow(W_eng/3000, 2/3)
    r.engine1.y = 6*lam
    r.engine2.y = 6*lam
    r.wingrect.scaley = 1-lam
    r.wingrect.y = -6 + 5*lam
    r.wingtaper.scaley = lam
    r.wingtaper.y = 5*lam
    </script>



.. raw:: html

    <script>var jswidget_0 = {storage: [], n:2, ranges: {}, after: document.getElementById('jswidget_0-after').innerText, bases: [1] }</script>



.. raw:: html

    <div id='jswidget_0_container'></div><style>#jswidget_0_container td {text-align: right; border: none !important;}
    #jswidget_0_container tr {border: none !important;}
    #jswidget_0_container table {border: none !important;}
    </style>



.. raw:: html

    <script>jswidget_0.ranges.var0 = [20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50]
    jswidget_0.bases.push(11)</script>



.. raw:: html

    <script>jswidget_0.ranges.var1 = [1000000.0, 2000000.0, 3000000.0, 4000000.0, 5000000.0, 6000000.0, 7000000.0, 8000000.0, 9000000.0, 10000000.0]
    jswidget_0.bases.push(110)</script>



.. raw:: html

    <script> jswidget_0.storage = ['\nvar W_eng = 8405.20714131,\n    lam = 0.450000000221\n\nr.shearinner.scalex = 1-0.00272112361846*10\nr.shearinner.scaley = 1-0.000222309606558*100\nr.airfoil.scaley = 0.143431169598/0.13\nr.fuse.scalex = 21720.7479109/24000\nr.wing.scalex = 30.9614116167/2/14\nr.wing.scaley = 4.58659372509*1.21\n', '\nvar W_eng = 6204.47129954,\n    lam = 0.450000001248\n\nr.shearinner.scalex = 1-0.00501498376518*10\nr.shearinner.scaley = 1-0.000343891069682*100\nr.airfoil.scaley = 0.146860187222/0.13\nr.fuse.scalex = 21353.3688207/24000\nr.wing.scalex = 28.6630507638/2/14\nr.wing.scaley = 3.4982978607*1.21\n', '\nvar W_eng = 4932.2959563,\n    lam = 0.45000000593\n\nr.shearinner.scalex = 1-0.00824168403638*10\nr.shearinner.scaley = 1-0.000494030333104*100\nr.airfoil.scaley = 0.150000000037/0.13\nr.fuse.scalex = 21152.2372887/24000\nr.wing.scalex = 26.7666742528/2/14\nr.wing.scaley = 2.81590881407*1.21\n', '\nvar W_eng = 4084.17744084,\n    lam = 0.450000002504\n\nr.shearinner.scalex = 1-0.0127258192678*10\nr.shearinner.scaley = 1-0.000678505573421*100\nr.airfoil.scaley = 0.149999003373/0.13\nr.fuse.scalex = 21030.1197817/24000\nr.wing.scalex = 24.9798866604/2/14\nr.wing.scaley = 2.36091347725*1.21\n', '\nvar W_eng = 3494.05170457,\n    lam = 0.450000000088\n\nr.shearinner.scalex = 1-0.0184707545514*10\nr.shearinner.scaley = 1-0.000893786510355*100\nr.airfoil.scaley = 0.149999999998/0.13\nr.fuse.scalex = 20952.6738911/24000\nr.wing.scalex = 23.3763467454/2/14\nr.wing.scaley = 2.03123094306*1.21\n', '\nvar W_eng = 3063.87102908,\n    lam = 0.450000000051\n\nr.shearinner.scalex = 1-0.0255356030944*10\nr.shearinner.scaley = 1-0.00113859322773*100\nr.airfoil.scaley = 0.15/0.13\nr.fuse.scalex = 20903.5213517/24000\nr.wing.scalex = 21.9102643801/2/14\nr.wing.scaley = 1.78282568215*1.21\n', '\nvar W_eng = 2739.78661319,\n    lam = 0.450000003839\n\nr.shearinner.scalex = 1-0.0339285270686*10\nr.shearinner.scaley = 1-0.00141067664328*100\nr.airfoil.scaley = 0.150000000048/0.13\nr.fuse.scalex = 20873.8769977/24000\nr.wing.scalex = 20.5532201203/2/14\nr.wing.scaley = 1.59020176616*1.21\n', '\nvar W_eng = 2489.72645506,\n    lam = 0.450000002337\n\nr.shearinner.scalex = 1-0.0436001480801*10\nr.shearinner.scaley = 1-0.00170684245539*100\nr.airfoil.scaley = 0.15000000002/0.13\nr.fuse.scalex = 20858.606759/24000\nr.wing.scalex = 19.2857859297/2/14\nr.wing.scaley = 1.43755610028*1.21\n', '\nvar W_eng = 2308.73897545,\n    lam = 0.450000003833\n\nr.shearinner.scalex = 1-0.053451351004*10\nr.shearinner.scaley = 1-0.0019948481008*100\nr.airfoil.scaley = 0.149999998248/0.13\nr.fuse.scalex = 20983.2466575/24000\nr.wing.scalex = 18.1949053015/2/14\nr.wing.scaley = 1.32428089408*1.21\n', '\nvar W_eng = 2307.54363499,\n    lam = 0.450000001009\n\nr.shearinner.scalex = 1-0.0535272899968*10\nr.shearinner.scaley = 1-0.00199702477187*100\nr.airfoil.scaley = 0.149999999994/0.13\nr.fuse.scalex = 22441.6484805/24000\nr.wing.scalex = 18.1870294178/2/14\nr.wing.scaley = 1.32352257272*1.21\n', '\nvar W_eng = 2307.54342571,\n    lam = 0.450000000513\n\nr.shearinner.scalex = 1-0.0535272937705*10\nr.shearinner.scaley = 1-0.00199702441791*100\nr.airfoil.scaley = 0.149999999999/0.13\nr.fuse.scalex = 23761.6864158/24000\nr.wing.scalex = 18.1870288935/2/14\nr.wing.scaley = 1.32352252727*1.21\n', '\nvar W_eng = 9147.94017688,\n    lam = 0.450000000154\n\nr.shearinner.scalex = 1-0.00250318353096*10\nr.shearinner.scaley = 1-0.000206224893337*100\nr.airfoil.scaley = 0.14382979484/0.13\nr.fuse.scalex = 24189.9396072/24000\nr.wing.scalex = 32.3582070069/2/14\nr.wing.scaley = 4.818148172*1.21\n', '\nvar W_eng = 6595.65348716,\n    lam = 0.450000000219\n\nr.shearinner.scalex = 1-0.00471394171944*10\nr.shearinner.scaley = 1-0.000323780120824*100\nr.airfoil.scaley = 0.146284115861/0.13\nr.fuse.scalex = 23308.41523/24000\nr.wing.scalex = 29.6752245221/2/14\nr.wing.scaley = 3.63967582205*1.21\n', '\nvar W_eng = 5187.54302319,\n    lam = 0.45000000128\n\nr.shearinner.scalex = 1-0.00777801778137*10\nr.shearinner.scaley = 1-0.00046734743579*100\nr.airfoil.scaley = 0.149999988317/0.13\nr.fuse.scalex = 22844.8094151/24000\nr.wing.scalex = 27.625541007/2/14\nr.wing.scaley = 2.91020107538*1.21\n', '\nvar W_eng = 4264.59405117,\n    lam = 0.450000002761\n\nr.shearinner.scalex = 1-0.0120671621851*10\nr.shearinner.scaley = 1-0.000644507063945*100\nr.airfoil.scaley = 0.149997600286/0.13\nr.fuse.scalex = 22567.9911757/24000\nr.wing.scalex = 25.7207368623/2/14\nr.wing.scaley = 2.43160400307*1.21\n', '\nvar W_eng = 3630.1945235,\n    lam = 0.450000002628\n\nr.shearinner.scalex = 1-0.0175691961813*10\nr.shearinner.scaley = 1-0.000851374046294*100\nr.airfoil.scaley = 0.149999998734/0.13\nr.fuse.scalex = 22393.3693463/24000\nr.wing.scalex = 24.0367934982/2/14\nr.wing.scaley = 2.08733649149*1.21\n', '\nvar W_eng = 3171.46830467,\n    lam = 0.450000000064\n\nr.shearinner.scalex = 1-0.0243410826927*10\nr.shearinner.scaley = 1-0.00108672296022*100\nr.airfoil.scaley = 0.15/0.13\nr.fuse.scalex = 22281.9815733/24000\nr.wing.scalex = 22.5131835064/2/14\nr.wing.scaley = 1.82920437703*1.21\n', '\nvar W_eng = 2827.82626853,\n    lam = 0.450000000868\n\nr.shearinner.scalex = 1-0.0323905711527*10\nr.shearinner.scaley = 1-0.00134839014671*100\nr.airfoil.scaley = 0.150000000013/0.13\nr.fuse.scalex = 22213.4136611/24000\nr.wing.scalex = 21.1139528385/2/14\nr.wing.scaley = 1.62971639488*1.21\n', '\nvar W_eng = 2563.73418363,\n    lam = 0.450000000364\n\nr.shearinner.scalex = 1-0.0416712796464*10\nr.shearinner.scaley = 1-0.00163332372714*100\nr.airfoil.scaley = 0.150000000005/0.13\nr.fuse.scalex = 22175.9197176/24000\nr.wing.scalex = 19.8153093041/2/14\nr.wing.scaley = 1.47201252916*1.21\n', '\nvar W_eng = 2357.0157795,\n    lam = 0.450000000151\n\nr.shearinner.scalex = 1-0.0520747237194*10\nr.shearinner.scaley = 1-0.00193760998239*100\nr.airfoil.scaley = 0.149999999993/0.13\nr.fuse.scalex = 22162.5072915/24000\nr.wing.scalex = 18.6006430039/2/14\nr.wing.scaley = 1.34517760725*1.21\n', '\nvar W_eng = 2323.84136588,\n    lam = 0.450000003594\n\nr.shearinner.scalex = 1-0.0541079068855*10\nr.shearinner.scaley = 1-0.00199562181473*100\nr.airfoil.scaley = 0.149999999983/0.13\nr.fuse.scalex = 23402.3802806/24000\nr.wing.scalex = 18.3842971084/2/14\nr.wing.scaley = 1.32448142215*1.21\n', '\nvar W_eng = 2323.83998735,\n    lam = 0.450000001105\n\nr.shearinner.scalex = 1-0.0541079377665*10\nr.shearinner.scaley = 1-0.00199561949822*100\nr.airfoil.scaley = 0.149999999955/0.13\nr.fuse.scalex = 24808.6404667/24000\nr.wing.scalex = 18.3842931729/2/14\nr.wing.scaley = 1.32448107538*1.21\n', '\nvar W_eng = 10038.478386,\n    lam = 0.45000000156\n\nr.shearinner.scalex = 1-0.0022865261785*10\nr.shearinner.scaley = 1-0.000190367482378*100\nr.airfoil.scaley = 0.144450751503/0.13\nr.fuse.scalex = 27119.0024372/24000\nr.wing.scalex = 33.9278314859/2/14\nr.wing.scaley = 5.08093240674*1.21\n', '\nvar W_eng = 7037.37687706,\n    lam = 0.450000000187\n\nr.shearinner.scalex = 1-0.00441765885385*10\nr.shearinner.scaley = 1-0.000304192678775*100\nr.airfoil.scaley = 0.145844007988/0.13\nr.fuse.scalex = 25505.6411912/24000\nr.wing.scalex = 30.7690056964/2/14\nr.wing.scaley = 3.79226619318*1.21\n', '\nvar W_eng = 5466.09035752,\n    lam = 0.450000002126\n\nr.shearinner.scalex = 1-0.00733403196006*10\nr.shearinner.scaley = 1-0.000441755100118*100\nr.airfoil.scaley = 0.14997758954/0.13\nr.fuse.scalex = 24698.5795677/24000\nr.wing.scalex = 28.5293881891/2/14\nr.wing.scaley = 3.0103094531*1.21\n', '\nvar W_eng = 4458.3766312,\n    lam = 0.450000000445\n\nr.shearinner.scalex = 1-0.0114372980704*10\nr.shearinner.scaley = 1-0.000611957113035*100\nr.airfoil.scaley = 0.149999723617/0.13\nr.fuse.scalex = 24226.4811978/24000\nr.wing.scalex = 26.4929943608/2/14\nr.wing.scaley = 2.5055276855*1.21\n', '\nvar W_eng = 3774.78390015,\n    lam = 0.450000000313\n\nr.shearinner.scalex = 1-0.0167099537726*10\nr.shearinner.scaley = 1-0.000810866421453*100\nr.airfoil.scaley = 0.149999991639/0.13\nr.fuse.scalex = 23930.9765294/24000\nr.wing.scalex = 24.7198725564/2/14\nr.wing.scaley = 2.14548410184*1.21\n', '\nvar W_eng = 3284.83610568,\n    lam = 0.450000000052\n\nr.shearinner.scalex = 1-0.0232055964763*10\nr.shearinner.scaley = 1-0.00103727732469*100\nr.airfoil.scaley = 0.15/0.13\nr.fuse.scalex = 23742.0223936/24000\nr.wing.scalex = 23.1331372409/2/14\nr.wing.scaley = 1.87694953593*1.21\n', '\nvar W_eng = 2920.02772866,\n    lam = 0.450000000516\n\nr.shearinner.scalex = 1-0.0309321597929*10\nr.shearinner.scaley = 1-0.00128911983906*100\nr.airfoil.scaley = 0.15000000001/0.13\nr.fuse.scalex = 23623.8696462/24000\nr.wing.scalex = 21.6877278452/2/14\nr.wing.scaley = 1.67018279582*1.21\n', '\nvar W_eng = 2640.86607879,\n    lam = 0.450000000101\n\nr.shearinner.scalex = 1-0.0398462927114*10\nr.shearinner.scaley = 1-0.00156347979678*100\nr.airfoil.scaley = 0.150000000003/0.13\nr.fuse.scalex = 23556.3059234/24000\nr.wing.scalex = 20.3547954151/2/14\nr.wing.scaley = 1.50714783846*1.21\n', '\nvar W_eng = 2422.98523812,\n    lam = 0.450000002757\n\nr.shearinner.scalex = 1-0.0498475133803*10\nr.shearinner.scaley = 1-0.00185665748832*100\nr.airfoil.scaley = 0.15000000005/0.13\nr.fuse.scalex = 23527.1569095/24000\nr.wing.scalex = 19.1145428657/2/14\nr.wing.scaley = 1.37625471208*1.21\n', '\nvar W_eng = 2340.13507943,\n    lam = 0.450000000523\n\nr.shearinner.scalex = 1-0.0546848342754*10\nr.shearinner.scaley = 1-0.00199422012948*100\nr.airfoil.scaley = 0.150000000001/0.13\nr.fuse.scalex = 24342.0163804/24000\nr.wing.scalex = 18.5803204121/2/14\nr.wing.scaley = 1.325437725*1.21\n', '\nvar W_eng = 2340.13409326,\n    lam = 0.450000000919\n\nr.shearinner.scalex = 1-0.0546849301295*10\nr.shearinner.scaley = 1-0.0019942246304*100\nr.airfoil.scaley = 0.149999999808/0.13\nr.fuse.scalex = 25710.0105988/24000\nr.wing.scalex = 18.5803101781/2/14\nr.wing.scaley = 1.32543679023*1.21\n', '\nvar W_eng = 11136.7927512,\n    lam = 0.450000000031\n\nr.shearinner.scalex = 1-0.00206885133612*10\nr.shearinner.scaley = 1-0.00017457341459*100\nr.airfoil.scaley = 0.145366218406/0.13\nr.fuse.scalex = 30683.1579898/24000\nr.wing.scalex = 35.7281758707/2/14\nr.wing.scaley = 5.38617220424*1.21\n', '\nvar W_eng = 7541.54549342,\n    lam = 0.450000000851\n\nr.shearinner.scalex = 1-0.00412487835314*10\nr.shearinner.scaley = 1-0.00028504464182*100\nr.airfoil.scaley = 0.145558610488/0.13\nr.fuse.scalex = 27998.8697751/24000\nr.wing.scalex = 31.9603467276/2/14\nr.wing.scaley = 3.95834509059*1.21\n', '\nvar W_eng = 5772.26309751,\n    lam = 0.450000000329\n\nr.shearinner.scalex = 1-0.00690591615938*10\nr.shearinner.scaley = 1-0.000417102339806*100\nr.airfoil.scaley = 0.149977362875/0.13\nr.fuse.scalex = 26739.2272171/24000\nr.wing.scalex = 29.4866718164/2/14\nr.wing.scaley = 3.11685932193*1.21\n', '\nvar W_eng = 4667.14783357,\n    lam = 0.450000036791\n\nr.shearinner.scalex = 1-0.0108344832815*10\nr.shearinner.scaley = 1-0.000580762221105*100\nr.airfoil.scaley = 0.149994780185/0.13\nr.fuse.scalex = 26020.7710279/24000\nr.wing.scalex = 27.2995037153/2/14\nr.wing.scaley = 2.58312147678*1.21\n', '\nvar W_eng = 3928.7180416,\n    lam = 0.450000000184\n\nr.shearinner.scalex = 1-0.0158895823713*10\nr.shearinner.scaley = 1-0.00077212164377*100\nr.airfoil.scaley = 0.149999701951/0.13\nr.fuse.scalex = 25575.6907283/24000\nr.wing.scalex = 25.4276783021/2/14\nr.wing.scaley = 2.20588483097*1.21\n', '\nvar W_eng = 3404.50538954,\n    lam = 0.450000000039\n\nr.shearinner.scalex = 1-0.0221242359692*10\nr.shearinner.scaley = 1-0.000990072517299*100\nr.airfoil.scaley = 0.15/0.13\nr.fuse.scalex = 25291.1093416/24000\nr.wing.scalex = 23.7715929159/2/14\nr.wing.scaley = 1.92618981098*1.21\n', '\nvar W_eng = 3016.73220202,\n    lam = 0.450000000193\n\nr.shearinner.scalex = 1-0.0295464684524*10\nr.shearinner.scaley = 1-0.00123263053389*100\nr.airfoil.scaley = 0.150000000008/0.13\nr.fuse.scalex = 25111.0643873/24000\nr.wing.scalex = 22.2756618936/2/14\nr.wing.scaley = 1.71168806627*1.21\n', '\nvar W_eng = 2721.358552,\n    lam = 0.450000004217\n\nr.shearinner.scalex = 1-0.0381159276103*10\nr.shearinner.scaley = 1-0.00149701932572*100\nr.airfoil.scaley = 0.150000000102/0.13\nr.fuse.scalex = 25004.5260525/24000\nr.wing.scalex = 20.9051698356/2/14\nr.wing.scaley = 1.54302627049*1.21\n', '\nvar W_eng = 2491.5399966,\n    lam = 0.450000000296\n\nr.shearinner.scalex = 1-0.0477396200621*10\nr.shearinner.scaley = 1-0.00177972306414*100\nr.airfoil.scaley = 0.150000000001/0.13\nr.fuse.scalex = 24952.8732576/24000\nr.wing.scalex = 19.6366865973/2/14\nr.wing.scaley = 1.40787132061*1.21\n', '\nvar W_eng = 2356.4532658,\n    lam = 0.450000000077\n\nr.shearinner.scalex = 1-0.0552570318656*10\nr.shearinner.scaley = 1-0.00199279770591*100\nr.airfoil.scaley = 0.149999999985/0.13\nr.fuse.scalex = 25347.3249428/24000\nr.wing.scalex = 18.7752774553/2/14\nr.wing.scaley = 1.32640340805*1.21\n', '\nvar W_eng = 2356.43238131,\n    lam = 0.450000000693\n\nr.shearinner.scalex = 1-0.0552583679953*10\nr.shearinner.scaley = 1-0.0019928363716*100\nr.airfoil.scaley = 0.149999999875/0.13\nr.fuse.scalex = 26688.0551698/24000\nr.wing.scalex = 18.7751299047/2/14\nr.wing.scaley = 1.32639032544*1.21\n', '\nvar W_eng = 12547.1678304,\n    lam = 0.450000000074\n\nr.shearinner.scalex = 1-0.00184699121626*10\nr.shearinner.scaley = 1-0.000158608992209*100\nr.airfoil.scaley = 0.146671472993/0.13\nr.fuse.scalex = 35183.1706607/24000\nr.wing.scalex = 37.8543123231/2/14\nr.wing.scaley = 5.75315912299*1.21\n', '\nvar W_eng = 8124.31531088,\n    lam = 0.45000000846\n\nr.shearinner.scalex = 1-0.00383476936617*10\nr.shearinner.scaley = 1-0.00026626110904*100\nr.airfoil.scaley = 0.145428638533/0.13\nr.fuse.scalex = 30861.8474333/24000\nr.wing.scalex = 33.2692621252/2/14\nr.wing.scaley = 4.14117390619*1.21\n', '\nvar W_eng = 6108.80785943,\n    lam = 0.45000000276\n\nr.shearinner.scalex = 1-0.00650002922033*10\nr.shearinner.scaley = 1-0.000393523484462*100\nr.airfoil.scaley = 0.149800361916/0.13\nr.fuse.scalex = 28998.8589899/24000\nr.wing.scalex = 30.4945078751/2/14\nr.wing.scaley = 3.23199743033*1.21\n', '\nvar W_eng = 4893.07559413,\n    lam = 0.450000014555\n\nr.shearinner.scalex = 1-0.0102556330832*10\nr.shearinner.scaley = 1-0.000550789848368*100\nr.airfoil.scaley = 0.149995832004/0.13\nr.fuse.scalex = 27968.9086754/24000\nr.wing.scalex = 28.1448626123/2/14\nr.wing.scaley = 2.66477918997*1.21\n', '\nvar W_eng = 4093.03789667,\n    lam = 0.450000000255\n\nr.shearinner.scalex = 1-0.0151049401021*10\nr.shearinner.scaley = 1-0.000735007822199*100\nr.airfoil.scaley = 0.149998972949/0.13\nr.fuse.scalex = 27339.2816311/24000\nr.wing.scalex = 26.1625874636/2/14\nr.wing.scaley = 2.26877805775*1.21\n', '\nvar W_eng = 3531.07733485,\n    lam = 0.450000001278\n\nr.shearinner.scalex = 1-0.0210925943809*10\nr.shearinner.scaley = 1-0.000944942220751*100\nr.airfoil.scaley = 0.149999999985/0.13\nr.fuse.scalex = 26937.6764097/24000\nr.wing.scalex = 24.4301681914/2/14\nr.wing.scaley = 1.97706776027*1.21\n', '\nvar W_eng = 3118.31979769,\n    lam = 0.450000000105\n\nr.shearinner.scalex = 1-0.0282274296072*10\nr.shearinner.scaley = 1-0.00117871177732*100\nr.airfoil.scaley = 0.150000000005/0.13\nr.fuse.scalex = 26681.4653573/24000\nr.wing.scalex = 22.8789575668/2/14\nr.wing.scaley = 1.75432655084*1.21\n', '\nvar W_eng = 2805.46514214,\n    lam = 0.450000000495\n\nr.shearinner.scalex = 1-0.0364720412979*10\nr.shearinner.scaley = 1-0.00143366834733*100\nr.airfoil.scaley = 0.150000000001/0.13\nr.fuse.scalex = 26525.8021508/24000\nr.wing.scalex = 21.467399036/2/14\nr.wing.scaley = 1.57971531627*1.21\n', '\nvar W_eng = 2562.91762296,\n    lam = 0.450000008424\n\nr.shearinner.scalex = 1-0.0457385336982*10\nr.shearinner.scaley = 1-0.0017064527032*100\nr.airfoil.scaley = 0.150000000051/0.13\nr.fuse.scalex = 26444.3700708/24000\nr.wing.scalex = 20.1681756882/2/14\nr.wing.scaley = 1.44010424622*1.21\n', '\nvar W_eng = 2376.65382859,\n    lam = 0.450000009607\n\nr.shearinner.scalex = 1-0.0555873791908*10\nr.shearinner.scaley = 1-0.00198477468812*100\nr.airfoil.scaley = 0.149999996113/0.13\nr.fuse.scalex = 26461.8318642/24000\nr.wing.scalex = 18.995539248/2/14\nr.wing.scaley = 1.32969560187*1.21\n', '\nvar W_eng = 2372.74335112,\n    lam = 0.450000000633\n\nr.shearinner.scalex = 1-0.0558283160765*10\nr.shearinner.scaley = 1-0.00199145649239*100\nr.airfoil.scaley = 0.149999999976/0.13\nr.fuse.scalex = 27842.4649871/24000\nr.wing.scalex = 18.9688026593/2/14\nr.wing.scaley = 1.32734253021*1.21\n', '\nvar W_eng = 14479.1583832,\n    lam = 0.450000000012\n\nr.shearinner.scalex = 1-0.00161467603103*10\nr.shearinner.scaley = 1-0.000142035707324*100\nr.airfoil.scaley = 0.148585210149/0.13\nr.fuse.scalex = 41210.5950813/24000\nr.wing.scalex = 40.4892311832/2/14\nr.wing.scaley = 6.21912992128*1.21\n', '\nvar W_eng = 8809.73798307,\n    lam = 0.450000000265\n\nr.shearinner.scalex = 1-0.00354470295494*10\nr.shearinner.scaley = 1-0.000247700212956*100\nr.airfoil.scaley = 0.145518355718/0.13\nr.fuse.scalex = 34199.7329713/24000\nr.wing.scalex = 34.7271042437/2/14\nr.wing.scaley = 4.34493518418*1.21\n', '\nvar W_eng = 6478.75343538,\n    lam = 0.450000003356\n\nr.shearinner.scalex = 1-0.0061235710547*10\nr.shearinner.scaley = 1-0.000371176604072*100\nr.airfoil.scaley = 0.14917921318/0.13\nr.fuse.scalex = 31518.2642385/24000\nr.wing.scalex = 31.5469030596/2/14\nr.wing.scaley = 3.35869650792*1.21\n', '\nvar W_eng = 5138.62040189,\n    lam = 0.450000007822\n\nr.shearinner.scalex = 1-0.00969882864678*10\nr.shearinner.scaley = 1-0.000521949730119*100\nr.airfoil.scaley = 0.149999992336/0.13\nr.fuse.scalex = 30092.643343/24000\nr.wing.scalex = 29.0335900867/2/14\nr.wing.scaley = 2.75105786679*1.21\n', '\nvar W_eng = 4268.96601947,\n    lam = 0.450000000243\n\nr.shearinner.scalex = 1-0.0143530774543*10\nr.shearinner.scaley = 1-0.000699401724388*100\nr.airfoil.scaley = 0.149998542871/0.13\nr.fuse.scalex = 29235.4354491/24000\nr.wing.scalex = 26.9273579257/2/14\nr.wing.scaley = 2.33443451282*1.21\n', '\nvar W_eng = 3665.23642293,\n    lam = 0.450000000221\n\nr.shearinner.scalex = 1-0.0201066945403*10\nr.shearinner.scaley = 1-0.000901732874615*100\nr.airfoil.scaley = 0.149999999994/0.13\nr.fuse.scalex = 28691.3077684/24000\nr.wing.scalex = 25.1106611904/2/14\nr.wing.scaley = 2.02974249937*1.21\n', '\nvar W_eng = 3225.21615389,\n    lam = 0.450000002801\n\nr.shearinner.scalex = 1-0.0269696206481*10\nr.shearinner.scaley = 1-0.00112717465403*100\nr.airfoil.scaley = 0.150000000079/0.13\nr.fuse.scalex = 28342.2922396/24000\nr.wing.scalex = 23.4989191204/2/14\nr.wing.scaley = 1.79820118901*1.21\n', '\nvar W_eng = 2893.4854386,\n    lam = 0.450000016233\n\nr.shearinner.scalex = 1-0.0349073240367*10\nr.shearinner.scaley = 1-0.00137321442945*100\nr.airfoil.scaley = 0.150000000242/0.13\nr.fuse.scalex = 28125.8902971/24000\nr.wing.scalex = 22.042526684/2/14\nr.wing.scaley = 1.61728931124*1.21\n', '\nvar W_eng = 2637.21746132,\n    lam = 0.450000004815\n\nr.shearinner.scalex = 1-0.0438388389117*10\nr.shearinner.scaley = 1-0.00163663825024*100\nr.airfoil.scaley = 0.150000000014/0.13\nr.fuse.scalex = 28005.7641865/24000\nr.wing.scalex = 20.7093649689/2/14\nr.wing.scaley = 1.47295877895*1.21\n', '\nvar W_eng = 2435.85200746,\n    lam = 0.450000000709\n\nr.shearinner.scalex = 1-0.0536300314437*10\nr.shearinner.scaley = 1-0.00191361854655*100\nr.airfoil.scaley = 0.14999999987/0.13\nr.fuse.scalex = 27960.0820224/24000\nr.wing.scalex = 19.4780377396/2/14\nr.wing.scaley = 1.35601181222*1.21\n', '\nvar W_eng = 2389.0750125,\n    lam = 0.450000001249\n\nr.shearinner.scalex = 1-0.0563947871655*10\nr.shearinner.scaley = 1-0.00199008117853*100\nr.airfoil.scaley = 0.149999999989/0.13\nr.fuse.scalex = 29028.0314139/24000\nr.wing.scalex = 19.161379215/2/14\nr.wing.scaley = 1.32829460864*1.21\n', '\nvar W_eng = 17449.0616118,\n    lam = 0.44999999999\n\nr.shearinner.scalex = 1-0.00137247575091*10\nr.shearinner.scaley = 1-0.000124472906935*100\nr.airfoil.scaley = 0.149999999994/0.13\nr.fuse.scalex = 50254.5239975/24000\nr.wing.scalex = 43.971804612/2/14\nr.wing.scaley = 6.89409580108*1.21\n', '\nvar W_eng = 9632.86897727,\n    lam = 0.450000000017\n\nr.shearinner.scalex = 1-0.00325359409754*10\nr.shearinner.scaley = 1-0.000229255989273*100\nr.airfoil.scaley = 0.145820783187/0.13\nr.fuse.scalex = 38169.9209526/24000\nr.wing.scalex = 36.3739463628/2/14\nr.wing.scaley = 4.57654167879*1.21\n', '\nvar W_eng = 6894.36088913,\n    lam = 0.450000001609\n\nr.shearinner.scalex = 1-0.00575792059426*10\nr.shearinner.scaley = 1-0.000349497394051*100\nr.airfoil.scaley = 0.148512126128/0.13\nr.fuse.scalex = 34350.4762394/24000\nr.wing.scalex = 32.6762630838/2/14\nr.wing.scaley = 3.49659692219*1.21\n', '\nvar W_eng = 5406.75434991,\n    lam = 0.450000002464\n\nr.shearinner.scalex = 1-0.00916247165387*10\nr.shearinner.scaley = 1-0.000494155420284*100\nr.airfoil.scaley = 0.150000000072/0.13\nr.fuse.scalex = 32418.4759133/24000\nr.wing.scalex = 29.9708796025/2/14\nr.wing.scaley = 2.84264824883*1.21\n', '\nvar W_eng = 4457.94379863,\n    lam = 0.450000000043\n\nr.shearinner.scalex = 1-0.013631254205*10\nr.shearinner.scaley = 1-0.000665188284703*100\nr.airfoil.scaley = 0.149999495102/0.13\nr.fuse.scalex = 31280.1968641/24000\nr.wing.scalex = 27.725180028/2/14\nr.wing.scaley = 2.40316626276*1.21\n', '\nvar W_eng = 3807.76691223,\n    lam = 0.450000000098\n\nr.shearinner.scalex = 1-0.0191629248285*10\nr.shearinner.scaley = 1-0.00086030619864*100\nr.airfoil.scaley = 0.149999999992/0.13\nr.fuse.scalex = 30562.9515716/24000\nr.wing.scalex = 25.8150819264/2/14\nr.wing.scaley = 2.08439302398*1.21\n', '\nvar W_eng = 3337.89683821,\n    lam = 0.450000000549\n\nr.shearinner.scalex = 1-0.0257681743583*10\nr.shearinner.scaley = 1-0.00107784200363*100\nr.airfoil.scaley = 0.150000000008/0.13\nr.fuse.scalex = 30101.6468262/24000\nr.wing.scalex = 24.1369710291/2/14\nr.wing.scaley = 1.84342485164*1.21\n', '\nvar W_eng = 2985.70691577,\n    lam = 0.450000007722\n\nr.shearinner.scalex = 1-0.0334154902947*10\nr.shearinner.scaley = 1-0.0013154099673*100\nr.airfoil.scaley = 0.149999999978/0.13\nr.fuse.scalex = 29811.1199547/24000\nr.wing.scalex = 22.6316192837/2/14\nr.wing.scaley = 1.65582279988*1.21\n', '\nvar W_eng = 2714.73005176,\n    lam = 0.450000004085\n\nr.shearinner.scalex = 1-0.0420295902823*10\nr.shearinner.scaley = 1-0.00156997024125*100\nr.airfoil.scaley = 0.149999999985/0.13\nr.fuse.scalex = 29642.7168426/24000\nr.wing.scalex = 21.2615149722/2/14\nr.wing.scaley = 1.50652649797*1.21\n', '\nvar W_eng = 2502.42452253,\n    lam = 0.450000002528\n\nr.shearinner.scalex = 1-0.051483048292*10\nr.shearinner.scaley = 1-0.00183780892737*100\nr.airfoil.scaley = 0.149999999874/0.13\nr.fuse.scalex = 29567.7642567/24000\nr.wing.scalex = 20.0022423814/2/14\nr.wing.scaley = 1.38578646262*1.21\n', '\nvar W_eng = 2405.46272982,\n    lam = 0.450000001389\n\nr.shearinner.scalex = 1-0.056956054909*10\nr.shearinner.scaley = 1-0.00198865962306*100\nr.airfoil.scaley = 0.149999999989/0.13\nr.fuse.scalex = 30244.9036877/24000\nr.wing.scalex = 19.3531103695/2/14\nr.wing.scaley = 1.32926438393*1.21\n', '\nvar W_eng = 24341.6508981,\n    lam = 0.45\n\nr.shearinner.scalex = 1-0.00105937298067*10\nr.shearinner.scaley = 1-0.000101496808121*100\nr.airfoil.scaley = 0.15/0.13\nr.fuse.scalex = 70153.5461842/24000\nr.wing.scalex = 50.2340445902/2/14\nr.wing.scaley = 8.31453800134*1.21\n', '\nvar W_eng = 10651.6336506,\n    lam = 0.450000000052\n\nr.shearinner.scalex = 1-0.0029579852582*10\nr.shearinner.scaley = 1-0.000210717264437*100\nr.airfoil.scaley = 0.146398640134/0.13\nr.fuse.scalex = 43025.1834846/24000\nr.wing.scalex = 38.2765162776/2/14\nr.wing.scaley = 4.84649305375*1.21\n', '\nvar W_eng = 7368.91836561,\n    lam = 0.450000000025\n\nr.shearinner.scalex = 1-0.00539451978416*10\nr.shearinner.scaley = 1-0.000328198756989*100\nr.airfoil.scaley = 0.147991142097/0.13\nr.fuse.scalex = 37566.5666427/24000\nr.wing.scalex = 33.9093292081/2/14\nr.wing.scaley = 3.64684409719*1.21\n', '\nvar W_eng = 5701.29636356,\n    lam = 0.450000000156\n\nr.shearinner.scalex = 1-0.00864435684951*10\nr.shearinner.scaley = 1-0.000467306763088*100\nr.airfoil.scaley = 0.149999999296/0.13\nr.fuse.scalex = 34979.162136/24000\nr.wing.scalex = 30.963711854/2/14\nr.wing.scaley = 2.94033283347*1.21\n', '\nvar W_eng = 4661.62150805,\n    lam = 0.450000000316\n\nr.shearinner.scalex = 1-0.0129374068899*10\nr.shearinner.scaley = 1-0.000632270314299*100\nr.airfoil.scaley = 0.149996299879/0.13\nr.fuse.scalex = 33492.5604104/24000\nr.wing.scalex = 28.5594010838/2/14\nr.wing.scaley = 2.47536930937*1.21\n', '\nvar W_eng = 3959.5735284,\n    lam = 0.450000002689\n\nr.shearinner.scalex = 1-0.0182579857697*10\nr.shearinner.scaley = 1-0.000820534235908*100\nr.airfoil.scaley = 0.149999999772/0.13\nr.fuse.scalex = 32565.2013992/24000\nr.wing.scalex = 26.5456975112/2/14\nr.wing.scaley = 2.14122227145*1.21\n', '\nvar W_eng = 3456.90904365,\n    lam = 0.450000011913\n\nr.shearinner.scalex = 1-0.0246186861051*10\nr.shearinner.scaley = 1-0.00103057269237*100\nr.airfoil.scaley = 0.150000000043/0.13\nr.fuse.scalex = 31968.6565239/24000\nr.wing.scalex = 24.7946838763/2/14\nr.wing.scaley = 1.89012304755*1.21\n', '\nvar W_eng = 3082.49646281,\n    lam = 0.450000010836\n\nr.shearinner.scalex = 1-0.031990678617*10\nr.shearinner.scaley = 1-0.00126008543435*100\nr.airfoil.scaley = 0.149999999866/0.13\nr.fuse.scalex = 31588.5567184/24000\nr.wing.scalex = 23.235872071/2/14\nr.wing.scaley = 1.69540237759*1.21\n', '\nvar W_eng = 2795.68301163,\n    lam = 0.450000004587\n\nr.shearinner.scalex = 1-0.0403039511913*10\nr.shearinner.scaley = 1-0.00150622486364*100\nr.airfoil.scaley = 0.149999999961/0.13\nr.fuse.scalex = 31360.8534296/24000\nr.wing.scalex = 21.825510963/2/14\nr.wing.scaley = 1.54086222963*1.21\n', '\nvar W_eng = 2571.50546727,\n    lam = 0.450000001902\n\nr.shearinner.scalex = 1-0.0494456526378*10\nr.shearinner.scaley = 1-0.00176561658404*100\nr.airfoil.scaley = 0.149999999966/0.13\nr.fuse.scalex = 31246.8946772/24000\nr.wing.scalex = 20.5345857877/2/14\nr.wing.scaley = 1.41604098767*1.21\n', '\nvar W_eng = 2421.93198709,\n    lam = 0.450000000414\n\nr.shearinner.scalex = 1-0.0575110216351*10\nr.shearinner.scaley = 1-0.00198715943739*100\nr.airfoil.scaley = 0.149999999894/0.13\nr.fuse.scalex = 31493.9316503/24000\nr.wing.scalex = 19.5441708926/2/14\nr.wing.scaley = 1.3302631666*1.21\n', '', '\nvar W_eng = 11970.2594054,\n    lam = 0.450000000006\n\nr.shearinner.scalex = 1-0.00265205733464*10\nr.shearinner.scaley = 1-0.000191738182394*100\nr.airfoil.scaley = 0.147368667828/0.13\nr.fuse.scalex = 49213.7943369/24000\nr.wing.scalex = 40.5504704256/2/14\nr.wing.scaley = 5.1731366828*1.21\n', '\nvar W_eng = 7917.58796578,\n    lam = 0.450000000018\n\nr.shearinner.scalex = 1-0.00503320105647*10\nr.shearinner.scaley = 1-0.000307220844089*100\nr.airfoil.scaley = 0.147583572972/0.13\nr.fuse.scalex = 41264.7804524/24000\nr.wing.scalex = 35.2662817767/2/14\nr.wing.scaley = 3.81269963276*1.21\n', '\nvar W_eng = 6026.76370408,\n    lam = 0.450000003189\n\nr.shearinner.scalex = 1-0.0081436670299*10\nr.shearinner.scaley = 1-0.000441338806585*100\nr.airfoil.scaley = 0.149976715788/0.13\nr.fuse.scalex = 37815.8612609/24000\nr.wing.scalex = 32.0190938265/2/14\nr.wing.scaley = 3.04523505369*1.21\n', '\nvar W_eng = 4882.1226473,\n    lam = 0.450000002235\n\nr.shearinner.scalex = 1-0.0122685295107*10\nr.shearinner.scaley = 1-0.000600532200643*100\nr.airfoil.scaley = 0.149998123213/0.13\nr.fuse.scalex = 35895.1960211/24000\nr.wing.scalex = 29.4347847194/2/14\nr.wing.scaley = 2.55143968584*1.21\n', '\nvar W_eng = 4121.70604682,\n    lam = 0.450000004507\n\nr.shearinner.scalex = 1-0.0173888465112*10\nr.shearinner.scaley = 1-0.000782296597258*100\nr.airfoil.scaley = 0.149999999983/0.13\nr.fuse.scalex = 34712.6411715/24000\nr.wing.scalex = 27.3050848902/2/14\nr.wing.scaley = 2.20046195865*1.21\n', '\nvar W_eng = 3582.86076181,\n    lam = 0.450000037318\n\nr.shearinner.scalex = 1-0.0235171426055*10\nr.shearinner.scaley = 1-0.000985201671851*100\nr.airfoil.scaley = 0.149999998522/0.13\nr.fuse.scalex = 33953.6844188/24000\nr.wing.scalex = 25.4737947575/2/14\nr.wing.scaley = 1.93843573861*1.21\n', '\nvar W_eng = 3184.2438798,\n    lam = 0.450000014229\n\nr.shearinner.scalex = 1-0.0306277022361*10\nr.shearinner.scaley = 1-0.00120706624518*100\nr.airfoil.scaley = 0.149999999426/0.13\nr.fuse.scalex = 33466.0653509/24000\nr.wing.scalex = 23.8565649301/2/14\nr.wing.scaley = 1.73612011034*1.21\n', '\nvar W_eng = 2880.34286499,\n    lam = 0.450000006084\n\nr.shearinner.scalex = 1-0.0386553753586*10\nr.shearinner.scaley = 1-0.00144519551177*100\nr.airfoil.scaley = 0.149999999957/0.13\nr.fuse.scalex = 33166.4598105/24000\nr.wing.scalex = 22.4023613771/2/14\nr.wing.scaley = 1.57603138058*1.21\n', '\nvar W_eng = 2643.57129505,\n    lam = 0.450000003398\n\nr.shearinner.scalex = 1-0.0474953178354*10\nr.shearinner.scaley = 1-0.00169639739893*100\nr.airfoil.scaley = 0.149999999984/0.13\nr.fuse.scalex = 33005.17721/24000\nr.wing.scalex = 21.0777831381/2/14\nr.wing.scaley = 1.44698815282*1.21\n', '\nvar W_eng = 2457.05438611,\n    lam = 0.450000006\n\nr.shearinner.scalex = 1-0.0569563333772*10\nr.shearinner.scaley = 1-0.0019558470054*100\nr.airfoil.scaley = 0.149999998916/0.13\nr.fuse.scalex = 32958.7805629/24000\nr.wing.scalex = 19.8657198853/2/14\nr.wing.scaley = 1.34194304475*1.21\n', '', '\nvar W_eng = 13808.6817547,\n    lam = 0.450000000018\n\nr.shearinner.scalex = 1-0.00232488408146*10\nr.shearinner.scaley = 1-0.000171663715877*100\nr.airfoil.scaley = 0.148942338733/0.13\nr.fuse.scalex = 57668.5398758/24000\nr.wing.scalex = 43.4277279206/2/14\nr.wing.scaley = 5.59439453245*1.21\n', '\nvar W_eng = 8563.85932841,\n    lam = 0.450000000187\n\nr.shearinner.scalex = 1-0.00467001343182*10\nr.shearinner.scaley = 1-0.000286385565355*100\nr.airfoil.scaley = 0.147358941652/0.13\nr.fuse.scalex = 45586.8789946/24000\nr.wing.scalex = 36.7823562075/2/14\nr.wing.scaley = 3.99810351396*1.21\n', '\nvar W_eng = 6389.86626049,\n    lam = 0.450000000281\n\nr.shearinner.scalex = 1-0.00765598289774*10\nr.shearinner.scaley = 1-0.000416094187539*100\nr.airfoil.scaley = 0.149979886501/0.13\nr.fuse.scalex = 40981.3306366/24000\nr.wing.scalex = 33.1503715041/2/14\nr.wing.scaley = 3.15835869218*1.21\n', '\nvar W_eng = 5121.86748762,\n    lam = 0.450000001382\n\nr.shearinner.scalex = 1-0.0116228566703*10\nr.shearinner.scaley = 1-0.000569886715481*100\nr.airfoil.scaley = 0.149998745615/0.13\nr.fuse.scalex = 38515.5426986/24000\nr.wing.scalex = 30.3560782009/2/14\nr.wing.scaley = 2.63193020433*1.21\n', '\nvar W_eng = 4295.39333433,\n    lam = 0.450000001114\n\nr.shearinner.scalex = 1-0.0165526967944*10\nr.shearinner.scaley = 1-0.000745479818357*100\nr.airfoil.scaley = 0.14999999981/0.13\nr.fuse.scalex = 37022.2933666/24000\nr.wing.scalex = 28.0961968086/2/14\nr.wing.scaley = 2.26237919787*1.21\n', '\nvar W_eng = 3716.42499538,\n    lam = 0.450000002431\n\nr.shearinner.scalex = 1-0.0224600059974*10\nr.shearinner.scaley = 1-0.000941579903541*100\nr.airfoil.scaley = 0.149999999605/0.13\nr.fuse.scalex = 36068.472397/24000\nr.wing.scalex = 26.1762247408/2/14\nr.wing.scaley = 1.98851498086*1.21\n', '\nvar W_eng = 3291.37453237,\n    lam = 0.450000002616\n\nr.shearinner.scalex = 1-0.0293219215373*10\nr.shearinner.scaley = 1-0.00115617690501*100\nr.airfoil.scaley = 0.15000000001/0.13\nr.fuse.scalex = 35452.4382985/24000\nr.wing.scalex = 24.4950799667/2/14\nr.wing.scaley = 1.77807576487*1.21\n', '\nvar W_eng = 2969.00284909,\n    lam = 0.450000010663\n\nr.shearinner.scalex = 1-0.0370780094672*10\nr.shearinner.scaley = 1-0.00138669542149*100\nr.airfoil.scaley = 0.149999999884/0.13\nr.fuse.scalex = 35066.4730467/24000\nr.wing.scalex = 22.9931371235/2/14\nr.wing.scaley = 1.61210394246*1.21\n', '\nvar W_eng = 2718.72141215,\n    lam = 0.450000004647\n\nr.shearinner.scalex = 1-0.045630563464*10\nr.shearinner.scaley = 1-0.00163008654588*100\nr.airfoil.scaley = 0.149999999982/0.13\nr.fuse.scalex = 34847.3909612/24000\nr.wing.scalex = 21.6320085188/2/14\nr.wing.scaley = 1.47861779671*1.21\n', '\nvar W_eng = 2521.13407763,\n    lam = 0.450000000165\n\nr.shearinner.scalex = 1-0.0548464077918*10\nr.shearinner.scaley = 1-0.00188301720186*100\nr.airfoil.scaley = 0.149999999988/0.13\nr.fuse.scalex = 34755.5291533/24000\nr.wing.scalex = 20.3855971879/2/14\nr.wing.scaley = 1.36962829433*1.21\n'] </script>



.. raw:: html

    <script id="jswidget_0_template" type="text/ractive"><table><tr><td>$V_{stall,max}$</td><input value="{{var0}}" type="range" min="0" max="10" step="1"><td><span id="jswidget_0-var0"></span></td></tr>
    <tr><td>$R_{min}$</td><input value="{{var1}}" type="range" min="0" max="9" step="1"><td><span id="jswidget_0-var1"></span></td></tr>
    </table></script>



.. raw:: html

    <script>$.getScript('http://cdn.ractivejs.org/latest/ractive.min.js', function() {
              jswidget_0.ractive = new Ractive({
              el: 'jswidget_0_container',
              template: '#jswidget_0_template',
              magic: true,
              data: {var0: 5, var1: 4, },
              onchange: function() {
                  var idxsum = 0
                  for (var i=0; i<jswidget_0.n; i++) {
                      varname = 'var'+i
                      idx = jswidget_0.ractive.data[varname]
                      document.getElementById("jswidget_0-"+varname).innerText = Math.round(100*jswidget_0.ranges[varname][idx])/100
                      idxsum += idx*jswidget_0.bases[i]
                  }
                  if (jswidget_0.storage[idxsum] === "") {
                    r.infeasibilitywarning = "Infeasible problem"
                  } else {
                    r.infeasibilitywarning = ""
                    eval(jswidget_0.storage[idxsum] + jswidget_0.after)
                  }
                }
            });
    
            MathJax.Hub.Typeset()
            jswidget_0.ractive.onchange()
    })</script>


This concludes the Box example. Try playing around with the sliders up
above until you're bored of this system; then check out one of the other
examples. Thanks for reading!

Import CSS for nbviewer
~~~~~~~~~~~~~~~~~~~~~~~

If you have a local iPython stylesheet installed, this will add it to
the iPython Notebook:

.. code:: python

    from IPython import utils
    from IPython.core.display import HTML
    import os
    def css_styling():
        """Load default custom.css file from ipython profile"""
        base = utils.path.get_ipython_dir()
        styles = "<style>\n%s\n</style>" % (open(os.path.join(base,'profile_default/static/custom/custom.css'),'r').read())
        return HTML(styles)
    css_styling()



.. raw:: html

    <style>
    @import url('http://fonts.googleapis.com/css?family=Crimson+Text');
    @import url('http://fonts.googleapis.com/css?family=Roboto');
    @import url('http://fonts.googleapis.com/css?family=Kameron');
    @import url('http://fonts.googleapis.com/css?family=Lato:200');
    @import url('http://fonts.googleapis.com/css?family=Lato:300');
    @import url('http://fonts.googleapis.com/css?family=Lato:400');
    @import url('http://fonts.googleapis.com/css?family=Source+Code+Pro');
    
    /* Change code font */
    pre {
        font-family: 'Source Code Pro', Consolas, monocco, monospace;
    }
    
    div.input_area {
        border-width: 0 0 0 1px;
        border-color: rgba(0,0,0,0.10);
        background: white;
        border-radius: 0;
    }
    
    div.text_cell {
        max-width: 105ex; /* instead of 100%, */
    }
    
    div.text_cell_render {
        font-family: Roboto;
        font-size: 12pt;
        line-height: 145%; /* added for some line spacing of text. */
    }
    
    div.text_cell_render h1,
    div.text_cell_render h2,
    div.text_cell_render h3,
    div.text_cell_render h4,
    div.text_cell_render h5,
    div.text_cell_render h6 {
        font-family: 'Roboto';
    }
    
    div.text_cell_render h1 {
        font-size: 24pt;
    }
    
    div.text_cell_render h2 {
        font-size: 18pt;
    }
    
    div.text_cell_render h3 {
        font-size: 14pt;
    }
    
    .rendered_html pre,
    .rendered_html code {
        font-size: medium;
    }
    
    .rendered_html ol {
        list-style:decimal;
        margin: 1em 2em;
    }
    
    .prompt {
        opacity: 0.6;
    }
    
    .prompt.input_prompt {
        color: #668;
        font-family: 'Source Code Pro', Consolas, monocco, monospace;
    }
    
    .prompt.out_prompt_overlay {
        font-family: 'Source Code Pro', Consolas, monocco, monospace;
    }
    
    .cell.command_mode.selected {
        border-color: rgba(0,0,0,0.1);
    }
    
    div.cell.selected {
        border-width: 0 0 0 1px;
        border-color: rgba(0,0,0,0.1);
        border-radius: 0;
    }
    
    div.output_scroll {
        -webkit-box-shadow: inset 0 2px 8px rgba(0,0,0,0.1);
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.1);
        border-radious: 2px;
    }
    
    #menubar .navbar-inner {
        background: #fff;
        -webkit-box-shadow: none;
        box-shadow: none;
        border-radius: 0;
        border: none;
        font-family: Roboto;
        font-weight: 400;
    }
    
    .navbar-fixed-top .navbar-inner,
    .navbar-static-top .navbar-inner {
        box-shadow: none;
        -webkit-box-shadow: none;
        border: none;
    }
    
    div#notebook_panel {
        box-shadow: none;
        -webkit-box-shadow: none;
        border-top: none;
    }
    
    div#notebook {
        border-top: 1px solid rgba(0,0,0,0.15);
    }
    
    #menubar .navbar .navbar-inner,
    .toolbar-inner {
        padding-left: 0;
        padding-right: 0;
    }
    
    #checkpoint_status,
    #autosave_status {
        color: rgba(0,0,0,0.5);
    }
    
    #header {
        font-family: Roboto;
    }
    
    #notebook_name {
        font-weight: 200;
    }
    
    /*
        This is a lazy fix, we *should* fix the
        background for each Bootstrap button type
    */
    #site * .btn {
        background: #fafafa;
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    
    </style>



