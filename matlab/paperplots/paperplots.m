function paperplots()

setenv('DYLD_LIBRARY_PATH');

addpath('../convex_fitting');
fs = 18;
%ms = 10;
xlab = @(s) xlabel(s, 'fontsize', fs);
ylab = @(s) ylabel(s, 'fontsize', fs);
zlab = @(s) zlabel(s, 'fontsize', fs);
leg = @(c,l) legend(c, 'fontsize', fs, 'location', l);

fnum = 1;
    function n = newplot()
        figure(fnum); clf(fnum);
        n = fnum;
        fnum = fnum + 1;
        set(gca, 'fontsize', fs);
    end

    function savefig(name)
        set(gca, 'fontsize', fs);
        dir = '$HOME/berkeley/research/aircraft_design/papers/aiaa_journal/figs/';
        reldir = '../../figs/';
        print(gcf, '-dpsc', [reldir, name, '.eps']);
        unix(['epstopdf ', dir, name, '.eps']);
    end

%breguet range
newplot();
z = linspace(0, log(3), 50); 
ff = exp(z) - 1;    %exact fuel fraction
ffest = [z; z+z.^2/2; z+z.^2/2+z.^3/6; z+z.^2/2+z.^3/6+z.^4/24];
plot(z, ff, 'linewidth', 2);
hold on;
plot(z, ffest(4,:), 'r-', 'linewidth', 1.25);
plot(z, ffest(3,:), 'r--');
a = axis();
a(2) = max(z);
axis(a);
xlab('gRD/(h_{fuel}\eta_0L)');
ylab('\theta_{fuel}');
leg({'exact', '4-term Taylor approx', '3-term Taylor approx'}, 'northwest');
savefig('breguet_color');
newplot();
plot(z, ff, 'k', 'linewidth', 1.25);
hold on;
plot(z, ffest(4,:), 'k-.', 'linewidth', 1.25);
plot(z, ffest(3,:), 'k--', 'linewidth', 1.25);
a = axis();
a(2) = max(z);
axis(a);
xlab('gRD/(h_{fuel}\eta_0L)');
ylab('\theta_{fuel}');
leg({'exact', '4-term Taylor approx', '3-term Taylor approx'}, 'northwest');
savefig('breguet_gray');

%takeoff distance
newplot();
xi = (.005:.005:.93)'; %drag to thrust ratio at takeoff
xiplot = (0.001:.02:.95)';
f = @(x) -log(1-x)./x;
pp =[log(1.044)/0.049
    0.296/0.049
    log(.0464)/2.88
    2.73/2.88
    0.049
    2.88];
plot(xiplot, f(xiplot), 'linewidth', 2);
hold on;
plot(xiplot, exp(implicit_softmax_affine(log(xiplot), pp))+1, 'r', 'linewidth', 1.25);
plot(xiplot, 1 + xiplot/2, 'k--')
leg({'-log(1-\xi)/\xi', '2-term fitted posynomial', '2,3, and 4 term Taylor approximations'},'northwest');
plot(xiplot, 1 + xiplot/2 + xiplot.^2/3, 'k--');
plot(xiplot, 1 + xiplot/2 + xiplot.^2/3 + xiplot.^3/4, 'k--');
plot(xiplot, f(xiplot), 'linewidth', 2);
plot(xiplot, exp(implicit_softmax_affine(log(xiplot), pp))+1, 'r', 'linewidth', 1.25);
xlab('\xi');
ylab('1+y');
a = axis;
a(2) = 0.9;
a(4) = 2.6;
axis(a);
savefig('xtofit_color');
newplot();
plot(xiplot, f(xiplot), 'k', 'linewidth', 1);
hold on;
plot(xiplot, exp(implicit_softmax_affine(log(xiplot), pp))+1, 'k-.', 'linewidth', 1);
plot(xiplot, 1 + xiplot/2, 'k--')
leg({'-log(1-\xi)/\xi', '2-term fitted posynomial', '2,3, and 4 term Taylor approximations'},'northwest');
plot(xiplot, 1 + xiplot/2 + xiplot.^2/3, 'k--');
plot(xiplot, 1 + xiplot/2 + xiplot.^2/3 + xiplot.^3/4, 'k--');
plot(xiplot, f(xiplot), 'k', 'linewidth', 1);
plot(xiplot, exp(implicit_softmax_affine(log(xiplot), pp))+1, 'k-.', 'linewidth', 1);
xlab('\xi');
ylab('1+y');
a = axis;
a(2) = 0.9;
a(4) = 2.6;
axis(a);
savefig('xtofit_gray');

%structural weight fraction
newplot();
L = (.01:.001:1)'; %lambda
f = @(x) (1+x+x.^2)./((1+x).^2);
p = 1+2*L;
loglog(L, f(L), 'linewidth', 2);
xlab('\lambda');
hold on;
loglog(L, (.86*p.^-2.38 + .14*p.^.56).^(1/3.94), 'r', 'linewidth', 1.25);
leg({'(1 + \lambda + \lambda^2)/(1+\lambda)^2', '2-term fitted posynomial'}, 'northeast');
savefig('nuapprox_color');

newplot();
loglog(L, f(L), 'k--', 'linewidth', 2);
xlab('\lambda');
hold on;
loglog(L, (.86*p.^-2.38 + .14*p.^.56).^(1/3.94), 'k', 'linewidth', 1);
leg({'(1 + \lambda + \lambda^2)/(1+\lambda)^2', '2-term fitted posynomial'}, 'northeast');
savefig('nuapprox_gray');

end
