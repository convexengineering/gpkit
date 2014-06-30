xi = (.005:.005:.93)'; %drag to thrust ratio at takeoff
xiplot = xi(xi <= 0.9);
f = @(x) -log(1-x)./x;
close all


%fit a model
addpath('../convex_fitting/');
Ks = 2;
fits = compare_fits(log(xi), log(f(xi)-1), Ks, 1, 500);
hold on
fits.softmax.params{1}
fits.implicit.params{1}
plot(xiplot, exp(implicit_softmax_affine(log(xiplot), fits.implicit.params{1}))+1, 'r');

%plotting code is in paperplots.m