clc
clear p;
load NACA24xx_fits;
tic;
p = gp(1000,200);

%constants
%model constants
p.const('omega') = 0.5;	%width of wing box as % of chord
p.const('f_wadd') = 2;
p.const('Aprop') = pi*.65^2;
p.const('eta_v') = 0.85;
p.const('eta_eng') = 0.40;

%operational constants
p.const('rho') = 0.55;   %25K'
p.const('rhosl') = 1.23;
p.const('mu') = 1.54e-5; %25K'

%engineering constants
p.const('g') = 9.8;
p.const('hfuel') = 42e6;

% %operational constraints (explorable design space)
p.const('Nult') = 4.0;
p.const('sigma_alum') = 250e6;  %270e6  6061-T6 w/ small safety buffer to factor in uncertainty about exact alloy
p.const('rho_alum') = 2700;
p.const('Ealum') = 70e9;
p.const('sigma_carbon') = 579e6;
p.const('rho_carbon') = 1750;
p.const('Ecarbon') = 45.5e9; %40e9
p.const('CLmax') = 1.5;
p.const('Rreq') = 5000e3;

%multi-fc vars
p.addvars({'V','CL','CD','CDfuse','CDp','CDi','T','W','Re', 'eta_prop', 'eta0', 'H_np'}, {'outbound', 'return', 'sprint'});
p.addvars({'Wfuel'}, {'outbound', 'return'});

%objective
p.add_objective_term(mono('Wfuel'), 'outbound');
p.add_objective_term(mono('Wfuel'), 'return');

%steady level flight
p.add_constraint(mec({'rho', 'V', 'CL', 'S', 'W'}, 2, [-1 -2 -1 -1 1], 'lift=weight'));
p.add_constraint(mono({'T', 'rho', 'V', 'CD', 'S'}, 1/2, [-1 1 2 1 1], 'thrust=drag'));
p.add_constraint(mono({'Wmto', 'rhosl', 'Vstall', 'CLmax', 'S'}, 2, [1 -1 -2 -1 -1], 'lift=weight landing'));
p.add_constraint(mono({'Psprint', 'T@sprint', 'V@sprint', 'eta0@sprint'}, 1, [-1 1 1 -1], 'sprint power'));

p.add_constraint(mono({'eta0', 'eta_eng', 'eta_prop'}, 1, [1 -1 -1], 'efficiency breakdown'));
p.add_constraint(posy({'T', 'rho', 'Aprop', 'V'}, [1 2], [0 0 0 0; 1 -1 -1 -2], mono('H_np'), 'eta_prop Helper'));
p.add_constraint(posy({'eta_prop', 'eta_v', 'H_np'}, [1/2 1/2], [1 -1 0; 1 -1 1/2], [], 'propulsive efficiency'));

%reynolds number
p.add_constraint(mec({'Re', 'rho', 'V', 'S', 'A', 'mu'}, 1, [-1 1 1 1/2 -1/2 -1], 'Reynolds number defn'));

%drag model
p.add_constraint(posy({'CDfuse', 'CDp', 'CDi'}, [1 1 1], eye(3), mono('CD'), 'drag breakdown'));
p.add_constraint(mono({'CDfuse', 'S'}, .05, [-1 -1], 'fuselage drag model'));
%p.add_constraint(mono({'CDp', 'k', 'Re', 'Sfrac'}, .074, [-1 1 -.2 1], 'wing profile drag model'));
p.add_constraint(NACA24xx.implicit.posy{4});
p.add_constraint(mono({'CDi', 'CL', 'e', 'A'}, 1/pi, [-1 2 -1 -1], 'induced drag model'));
p.add_constraint(mono('e', 1/.90, 1, 'spanwise efficiency'));

%weights
p.add_constraint(posy({'Wfixed', 'Wwing', 'Wpay'}, [1 1 1], eye(3), mono('Wdry'), 'dry weight breakdown'));
p.add_constraint(mono({'Wdry\wing', 'Wfixed'}, 1, [-1 1], 'dry weight no wing'));
%p.add_constraint(posy({'S','Nult','A','Wmto\wing','Wmto','tau'},[45.42 8.71e-5],[1 0 0 0 0 0; 1/2 1 3/2 1/2 1/2 -1],mono('Wwing'),'wing weight model'));
p.add_constraint(posy({'Wdry', 'Wfuel@return', 'Wfuel@outbound'}, [1 1 1], eye(3), mono('Wmto'), 'MTOW'));
p.add_constraint(posy({'Wdry\wing', 'Wfuel@return', 'Wfuel@outbound'}, [1 1 1], eye(3), mono('Wmto\wing'), 'MTOW no wing'));
p.add_constraint(mono({'W', 'Wdry'}, 1, [-1 1], 'final weight'), 'return');
p.add_constraint(posy({'W@return', 'Wfuel@return'}, [1 1], eye(2), mono('W@outbound'), 'intermed weight'));
p.add_constraint(mono({'W@sprint', 'W@outbound'}, 1, [-1 1], 'sprint weight def'));
p.add_constraint(mono('Wfixed', 1667*9.8, -1, 'fixed weight'));

%range
bre = posy({'R','g','T','hfuel','W','eta0'}, [1 1/2 1/6 1/24], repmat((1:4)',1,6).*repmat([1 1 1 -1 -1 -1],4,1), mono({'Wfuel', 'W'}, 1, [1 -1]), 'Breguet range eqn');
p.add_constraint(bre);
%p.add_constraint(mono('eta0', 1/.2, 1, 'overall efficiency'));

%wing
%p.add_constraint(mono({'Wwing', 'g', 'rho_alum', 'tcap', 'S', 'omega', 'f_wadd'}, 2, [-1 1 1 1 1 1 1], 'wing weight model'));
%p.add_constraint(mono({'Wwing', 'f_wadd', 'Wcap'}, 1, [-1 1 1], 'wing weight model'));
p.add_constraint(posy({'Wweb', 'Wcap'}, [1 1], eye(2), mono({'Wwing', 'f_wadd'}, 1, [1 -1])));
p.add_constraint(mono('tau', 1/.14, 1, 'wing thickness'));
p.add_constraint(mono('lambda', 0.4, -1, 'tip stall'));

%wing root stress constraints -- version described in wing_struct.pdf -- doesn't work because cr is decoupled (cancelled) from root sizing
%p.add_constraint(posy({'lambda'}, [0 2], [0; 1], mono({'Mr/cr', 'A', 'Wmto\wing'}, 24, [1 -1 -1]), 'root moment per root chord'));
%p.add_constraint(posy({'tau', 'tcap', 'Icap', 'omega'}, [.92 1], [1 2 0 0; 0 0 1 -1], mono({'tau', 'tcap'}, (.92^2)/2, [2 1]), 'root area mom. of inertia'));
%p.add_constraint(posy({'lambda'}, 1/3*[1 1 1], [0; 1; 2], mono('wingV'), 'wing volume'));
%p.add_constraint(mono({'sigma_alum', 'Icap', 'Wcap', 'tau', 'Nult', 'Mr/cr', 'rho_alum', 'g', 'Acap', 'S', 'A', 'wingV'}, 1/2, [-1 -1 -1 1 1 1 1 1 1 1/2 1/2 1], 'spar cap weight'));
%p.add_constraint(mono({'Acap', 'omega', 'tcap'}, 2, [-1 1 1], 'spar cap area'));

%wing root stress
p.add_constraint(posy({'lambda'}, [9.69162e-5, .0019683], [.111855; 1.6626], mono({'Mr', 'Wmto\wing', 'S', 'A'}, 4^8.9361, 8.9361*[1 -1 -1/2 -1/2]), 'wing root moment'));
p.add_constraint(posy({'lambda'}, [1 1], [0; 1], mono({'S', 'A', 'cr'}, 2, [1/2 -1/2 -1]), 'root chord'));
p.add_constraint(posy({'tcap', 'omega', 'cr', 'Sr'}, [2*.92 1], [2 1 1 0; 0 0 0 1], mono({'omega', 'cr', 'tau', 'tcap'}, .92^2, [1 2 1 1]), 'root section modulus')); %.92 is rms height of parabola from 1 to .75 (see N+3 paper A.196)
p.add_constraint(mono({'Sr', 'sigma_carbon', 'Nult', 'Mr'}, 1, [-1 -1 1 1], 'root stress limit'));
p.add_constraint(mono({'Wcap', 'rho_carbon', 'g', 'S', 'omega', 'tcap'}, 2, [-1 1 1 1 1 1], 'spar cap weight'));
p.add_constraint(posy({'lambda'}, [1 1], [0; 1], mono({'tweb', 'sigma_carbon', 'tau', 'S', 'A', 'Wmto\wing', 'Nult'}, .75, [1 1 1 1/2 -1/2 -1 -1]), 'spar web sizing'));
p.add_constraint(mono({'Wweb', 'rho_carbon', 'g', 'S', 'tweb', 'tau'}, 1, [-1 1 1 1 1 1], 'spar web weight'));

%mission requirements
p.add_constraint(mono({'R', 'Rreq'}, 1, [-1 1], 'range requirement'));
p.add_constraint(mono('Wpay', 500*9.8, -1, 'payload requirement'));
p.add_constraint(mono({'V@sprint'}, 170, -1, 'max speed requirement'));
p.add_constraint(mono({'Psprint'}, 1/750e3, 1, 'power limit'));
p.add_constraint(mono('Vstall', 1/38, 1, 'stall speed requirement'));

%solve
p.trim();
t_parse = toc;
tic;
res = p.solve('minimize echo(0)');
t_solve = toc;
tic;
p.print_report(res, 'print_sensitivities', true);
t_print = toc;

disp(' ');
disp([' Load time: ', num2str(t_parse), 's']);
disp(['Solve time: ', num2str(t_solve), 's']);
disp(['Print time: ', num2str(t_print), 's']);

