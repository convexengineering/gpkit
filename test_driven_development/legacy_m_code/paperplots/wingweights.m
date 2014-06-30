L = (.01:.001:1)'; %lambda
f = @(x) (1+x+x.^2)./((1+x).^2);
close all;


%fit a model
addpath('../convex_fitting/');
Ks = 2;
fits = compare_fits(log(1+2*L), log(f(L)), Ks, 1, 500);
loglog(L, f(L), 'linewidth', 2);
hold on
fits.softmax.params{1}
fits.implicit.params{1}
loglog(L, exp(softmax_affine(log(1+2*L), fits.softmax.params{1})), 'r');
p = 2*L + 1;
hold on;
h = loglog(L, (.86*p.^-2.38 + .14*p.^.56).^(1/3.94), 'g')