clear p;
load NACA24xx_fits;
tic;
p = gp(1000,200);

p.const('Nlift') = 6.0;
p.const('sigma_max') = 250e6; %aluminum 6061-T6 w/ small safety buffer to factor in uncertainty about exact alloy
p.const('sigma_max_shear') = 167e6; %aluminum
p.const('rho_alum') = 2700;
%p.const('Ealum') = 70e9;
p.const('g') = 9.8;
p.const('wbar') = 0.5;
p.const('rh') = 0.75;
p.const('f_wadd') = 2.0;
p.const('Wfixed') = 1500*9.8;
p.const('CLmax') = 1.5;
p.const('rho') = 0.91;    %3000m
p.const('rhosl') = 1.23;
p.const('mu') = 1.69e-5; %3000m
p.const('e') = 0.95;
p.const('Aprop') = 0.785; %pi*.5^2;
p.const('eta_v') = 0.85;
p.const('eta_eng') = 0.35;
p.const('Rreq') = 5000e3;
p.const('hfuel') = 42e6;

p.addvars({'V','CL','CD','CDfuse','CDp','CDi','T','W','Re', 'eta_i','eta_prop', 'eta0'}, {'outbound', 'return', 'sprint'});
p.addvars({'Wfuel', 'z_bre'}, {'outbound', 'return'});

p.add_objective_term(mono('Wfuel'), 'outbound');
p.add_objective_term(mono('Wfuel'), 'return');
%p.add_objective_term(mono('delta/b', 500, 1));

%steady level flight
p.add_constraint(mec({'rho', 'V', 'CL', 'S', 'W'}, 2, [-1 -2 -1 -1 1], 'lift=weight'));
p.add_constraint(mono({'T', 'rho', 'V', 'CD', 'S'}, 1/2, [-1 1 2 1 1], 'thrust=drag'));
p.add_constraint(mono({'Wmto', 'rhosl', 'Vstall', 'CLmax', 'S'}, 2, [1 -1 -2 -1 -1], 'landing lift=weight'));
p.add_constraint(mono({'Pmax', 'T@sprint', 'V@sprint', 'eta0@sprint'}, 1, [-1 1 1 -1], 'sprint power'));

% %reynolds number
p.add_constraint(mec({'Re', 'rho', 'V', 'S', 'A', 'mu'}, 1, [-1 1 1 1/2 -1/2 -1], 'Reynolds number defn'));
% 
% %drag model
p.add_constraint(posy({'CDfuse', 'CDp', 'CDi'}, [1 1 1], eye(3), mono('CD'), 'drag breakdown'));
p.add_constraint(mono({'CDfuse', 'S'}, .05, [-1 -1], 'fuselage drag model'));
p.add_constraint(NACA24xx.implicit.posy{4});
p.add_constraint(mono({'CDi', 'CL', 'e', 'A'}, 1/pi, [-1 2 -1 -1], 'induced drag model'));
% p.add_constraint(mono('e', 1/.90, 1, 'spanwise efficiency'));

%efficiencies
p.add_constraint(mono({'eta0', 'eta_eng', 'eta_prop'}, 1, [1 -1 -1], 'overall efficiency'));
p.add_constraint(mono({'eta_prop', 'eta_i', 'eta_v'}, 1, [1 -1 -1], 'propeller efficiency'));
p.add_constraint(posy({'eta_i', 'T', 'rho', 'V', 'Aprop'}, [1 1/2], [1 0 0 0 0; 2 1 -1 -2 -1], [], 'inviscid efficiency'));

%Breguet range eqn
%bre = posy({'R','g','T','hfuel','W','eta0'}, [1 1/2 1/6 1/24], repmat((1:4)',1,6).*repmat([1 1 1 -1 -1 -1],4,1), mono({'Wfuel', 'W'}, 1, [1 -1]), 'Breguet range eqn');
p.add_constraint(mono({'z_bre', 'g', 'R', 'T', 'hfuel', 'eta0', 'W'}, 1, [-1 1 1 1 -1 -1 -1], 'bre helper'));
p.add_constraint(posy({'z_bre'}, [1 1/2 1/6 1/24], [1; 2; 3; 4], mono({'Wfuel', 'W'}, 1, [1 -1]), 'Breguet range eqn'));

%weights
p.add_constraint(posy({'Wfixed', 'Wpay', 'Weng'}, [1 1 1], eye(3), mono('Wtilde'), 'non-wing operating weight'));
p.add_constraint(posy({'Wtilde', 'Wwing'}, [1 1], eye(2), mono('Wdry'), 'total operating weight'));
p.add_constraint(mono({'Weng', 'Pmax'}, 9.8*.0038, [-1 .803], 'engine weight model'));

%mission weight definitions
p.add_constraint(mono({'W@return', 'Wdry'}, 1, [-1 1], 'final weight'));
p.add_constraint(posy({'W@return', 'Wfuel@return'}, [1 1], eye(2), mono('W@outbound'), 'intermed weight'));
p.add_constraint(posy({'W@outbound', 'Wfuel@outbound'}, [1 1], eye(2), mono('Wmto'), 'MTOW'));
p.add_constraint(mono({'W@sprint', 'W@outbound'}, 1, [-1 1], 'sprint weight def'));

%wing structure
p.add_constraint(posy({'p'}, [1/2 1/2], [0; 1], mono('q'), 'wing q definition'));
p.add_constraint(mono('p', 1.9, -1, 'wing tip stall'));
p.add_constraint(mono({'Mr/cr', 'Wtilde', 'A', 'p'}, 1/24, [-1 1 1 1], 'wing root moment per chord'));
p.add_constraint(posy({'wbar','tau','tcap','Ibar'}, [0.92 1], [1 1 1 0; 0 0 -1 1], mono({'wbar', 'tau'}, .5*.92^2, [1 2]), 'wing area moment of inertia'));
p.add_constraint(mono({'Nlift', 'Mr/cr', 'A', 'q', 'tau', 'S', 'Ibar', 'sigma_max'}, 1/8, [1 1 1 2 1 -1 -1 -1], 'wing root stress limit'));
p.add_constraint(mono({'A','Wtilde','Nlift','q','tau','S','tweb','sigma_max_shear'}, 1/12, [1 1 1 2 -1 -1 -1 -1], 'wing root shear stress'));
p.add_constraint(posy({'p'}, [0.86 0.14], [-2.38; 0.56], mono('nu', 1, 3.94), 'wing nu definition'));
p.add_constraint(mono({'Wcap','rho_alum','g','wbar','tcap','S','nu','A'}, 8/3, [-1 1 1 1 1 3/2 1 -1/2], 'wing spar cap weight'));
p.add_constraint(mono({'Wweb','rho_alum','g','rh','tau','tweb','S','nu','A'}, 8/3, [-1 1 1 1 1 1 3/2 1 -1/2], 'wing shear web weight'));
%p.add_constraint(mono({'delta/b','A','Mr/cr','q','S','Ealum','Ibar'}, 1/64, [-1 2 1 3 -1 -1 -1], 'wing tip deflection'));
p.add_constraint(posy({'Wcap', 'Wweb'}, [1 1], eye(2), mono({'Wwing', 'f_wadd'}, 1, [1 -1]), 'wing weight'));
p.add_constraint(mono('tau', 1/.15, 1, 'wing thickness limit'));

%requirements
p.add_constraint(mono({'R', 'Rreq'}, 1, [-1 1], 'range requirement'));
p.add_constraint(mono({'Wpay', 'g'}, 500, [-1 1], 'payload requirement'));
p.add_constraint(mono('Vstall', 1/38, 1, 'stall speed requirement'));
p.add_constraint(mono({'V@sprint'}, 150, -1, 'max speed requirement'));
%p.add_constraint(mono({'Pmax'}, 1/1200e3, 1, 'power limit'));


%solve
p.trim();
tload = toc;
tic; 
res = p.solve('minimize echo(0)');
tsolve = toc;
tic; 
p.print_report(res, 'print_sensitivities', true);
tprint = toc;

disp(' ');
disp([' Load time: ', num2str(tload), 's']);
disp(['Solve time: ', num2str(tsolve), 's']);
disp(['Print time: ', num2str(tprint), 's']);


