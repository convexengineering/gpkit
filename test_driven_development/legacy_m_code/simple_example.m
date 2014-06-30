function [p, sols] = simple_example()

setenv('DYLD_LIBRARY_PATH');

tic();
clear p
p = gp(100,100);

%constants
p.const('rho') = 1.23;
p.const('mu') = 1.78e-5;
%p.const('V') = 40;
p.const('Sfrac') = 2.05;    %Swet/S
p.const('k') = 1.2;
p.const('tau') = 0.12;

%constants for sensitivity analysis
p.add_constraint(mono('CDA0', 0.0306, -1, 'CDA0 -- fuselage drag area', 'const'));
p.add_constraint(mono('e', 1/0.96, 1, 'e -- Oswald efficiency', 'const'));
p.add_constraint(mono('W0', 4940, -1, 'W0 -- fixed weight', 'const'));
p.add_constraint(mono('Nult', 2.5, -1, 'Nult -- ultimate load factor', 'const'));
p.add_constraint(mono('Vmin', 1/22.0, 1, 'landing speed', 'const'));
p.add_constraint(mono('CLmax', 1/2.0, 1, 'CLmax -- stall CL', 'const'));

%objective
p.add_objective_term(mono('D'));

%models

%steady level flight
p.add_constraint(mono({'W', 'rho', 'V', 'CL', 'S'}, 2, [1 -1 -2 -1 -1], 'CL definition', ''));
p.add_constraint(mono({'D', 'rho', 'V', 'CD', 'S'}, 1/2, [-1 1 2 1 1], 'CD definition'));

%drag models
p.add_constraint(posy({'CDfuse', 'CDp', 'CDi'}, [1 1 1], eye(3), mono('CD'), 'drag breakdown', ''));
p.add_constraint(mono({'CDfuse', 'CDA0', 'S'}, 1, [-1 1 -1], 'fuselage drag model'));
p.add_constraint(mono({'CDp', 'k', 'Cf', 'Sfrac'}, 1, [-1 1 1 1], 'wing profile drag model'));
p.add_constraint(mono({'CDi', 'CL', 'A', 'e'}, 1/pi, [-1 2 -1 -1], 'induced drag model'));
p.add_constraint(mono({'Cf', 'Re'}, 0.074, [-1 -.2], 'skin friction coefficient'));
p.add_constraint(mono({'Re', 'mu', 'A', 'rho', 'V', 'S'}, 1, [1 1 .5 -1 -1 -.5], 'Reynolds number definition'));

%weight models
p.add_constraint(posy({'W0', 'Wwing'}, [1 1], eye(2), mono('W'), 'weight breakdown'));
p.add_constraint(posy({'S', 'Nult', 'A', 'W0', 'W', 'tau'}, [45.42, 8.71e-5], [1 0 0 0 0 0; 1/2 1 3/2 1/2 1/2 -1], mono('Wwing'), 'wing weight model'));

%landing speed
p.add_constraint(mono({'CLmax', 'W', 'rho', 'Vmin', 'S'}, 2, [-1 1 -1 -2 -1], 'landing stall limit'));

%solve
p.trim();
tload = toc();
tic();
res = p.solve('minimize echo(0)');
tsolve = toc();
tic();
p.print_report(res);
tprint = toc();

tic();
nvmin = 31;
nv = 25;
%sweep out optimal velocities for each landing speed
sols = p.sweep_pareto('landing speed', [], logspace(log10(20/22), log10(40/22), nvmin));
close(1);
tsweep = toc();

fprintf('\n')
fprintf([' load time: ', num2str(tload), ' sec\n']);
fprintf(['solve time: ', num2str(tsolve), ' sec\n']);
fprintf(['print time: ', num2str(tprint), ' sec\n']);
fprintf(['sweep time: ', num2str(tsweep), ' sec\n']);

%now sweep up to higher vels as well for each landing speed
p.add_constraint(mono('V', 38, -1, 'cruise speed'));
Vmin = repmat(sols(p.vari('Vmin'),:), nv, 1);
%V = linspace(1,1.5,nv)'*sols(p.vari('V'),:);
V = zeros(size(Vmin));
for i = 1:size(V,2)
    V(:,i) = linspace(sols(p.vari('V'),i), 60, nv);
end
sols = nan*zeros(p.nvar, numel(V));
iv = find(strcmp(p.constraints, 'cruise speed'));
ivmin = find(strcmp(p.constraints, 'landing speed'));
vrows = find(p.map == iv);
vminrows = find(p.map == ivmin);
tic();
for i = 1:numel(V)
    p.c(vrows) = V(i);
    p.c(vminrows) = 1/Vmin(i);
    res = p.solve();
    if strcmp(res.sol.itr.solsta, 'OPTIMAL')
        sols(:,i) = exp(res.sol.itr.xx);
    end
end
tsurf = toc();
disp(['solve time was ', num2str(tsurf), ' sec total, or ', num2str(tsurf/numel(V)), ' sec on average']);

fs = 18;
%ms = 10;
xlab = @() xlabel('cruise speed V', 'fontsize', fs);
ylab = @() ylabel('landing speed V_{min}', 'fontsize', fs);
zlab = @(s) zlabel(s, 'fontsize', fs);

    function savefig(name)
        set(gca, 'fontsize', fs);
        dir = '$HOME/berkeley/research/aircraft_design/papers/aiaa_journal/figs/';
        reldir = '../figs/';
        print(gcf, '-dpsc', [reldir, name, '.eps']);
        unix(['epstopdf ', dir, name, '.eps']);
    end

    function h=plot2dPF(c)
        %also touches up axis limits
        hold on;
        a = axis;
        h=plot3(V(1,:), Vmin(1,:), a(5)*ones(1,size(V,2)), c);
        %plot3(V(1,:), Vmin(1,:), a(5)*ones(1,size(V,2)), 'k.', 'markersize', ms);
        a(3:4) = [min(min(Vmin)), max(max(Vmin))];
        axis(a);
        hold off;
    end

figure(2); clf(2);
set(gca, 'fontsize', fs);

graymap = .6*(1-colormap('gray'));
mesh(V, Vmin, reshape(sols(p.vari('D'),:), size(V)));
h=plot2dPF('b');
xlab(); ylab();
zlab('total drag D');
view(-60,30);
colormap winter;
savefig('simple_D_color');
colormap(graymap);
delete(h);
plot2dPF('k');
savefig('simple_D_gray');

mesh(V, Vmin, reshape(sols(p.vari('S'),:), size(V)));
h=plot2dPF('b');
xlab(); ylab();
zlab('wing area S');
view(-60,30);
colormap winter;
savefig('simple_S_color');
colormap(graymap);
delete(h);
plot2dPF('k');
savefig('simple_S_gray');

mesh(V, Vmin, reshape(sols(p.vari('A'),:), size(V)));
h=plot2dPF('b');
xlab(); ylab();
zlab('aspect ratio A');
view(-50,20);
colormap winter;
savefig('simple_A_color');
colormap(graymap);
delete(h);
plot2dPF('k');
savefig('simple_A_gray');

mesh(V, Vmin, reshape(sols(p.vari('Wwing'),:), size(V)));
h=plot2dPF('b');
xlab(); ylab();
zlab('wing weight W_{wing}');
view(-30,15);
colormap winter;
savefig('simple_Ww_color');
colormap(graymap);
delete(h);
plot2dPF('k');
savefig('simple_Ww_gray');

end