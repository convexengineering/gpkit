addpath('fitting');

load('profile_drag_data_NACA24xx');
y = log(data.cd);
x = log([data.cl, data.re, data.tau]);
Ks = 2:10;
fits = compare_fits(x, y, Ks, 15, 500);

figure(1); clf(1);
types = {'maxaffine', 'softmax','implicit'};
clrs = {'b', 'r', 'g', 'm'};
bestfits.Ks = Ks;
for i = 1:length(types)
    [rr, ii] = min(fits.(types{i}).resid, [], 2);
    plot(Ks, rr, [clrs{i}, '-']); hold on;
    bestfits.(types{i}).resid = rr;
    for j = 1:length(Ks)
        bestfits.(types{i}).params{j} = fits.(types{i}).params{j, ii(j)};
        bestfits.(types{i}).maxresid(j) = fits.(types{i}).maxresid(j,ii(j));
    end
end
legend(types);
for i = 1:length(types)
    plot(Ks, bestfits.(types{i}).resid, [clrs{i}, '.']);
end
figure(2); clf(2);
for i = 1:length(types)
    plot(Ks, bestfits.(types{i}).maxresid, [clrs{i}, '-']); hold on;
end
legend(types);
for i = 1:length(types)
    plot(Ks, bestfits.(types{i}).maxresid, [clrs{i}, '.']); hold on;
end

%convert to models
addpath('..');
for i = 1:length(Ks)
    k = Ks(i);
    params = bestfits.softmax.params{i};
    alpha = 1/params(end);
    ba = reshape(params(1:end-1), 4, k);
    bestfits.softmax.posy{i} = posy({'CL', 'Re', 'tau'}, exp(alpha*ba(1,:)), alpha*ba(2:end,:)', mono('CDp', 1, alpha), 'softmax wing profile drag model');
    params = bestfits.implicit.params{i};
    alpha = params(end-(k-1):end);
    amat = repmat(alpha, 1, 3);
    ba = reshape(params(1:end-k), 4, k);
    bestfits.implicit.posy{i} = posy({'CL', 'Re', 'tau', 'CDp'}, exp(alpha.*ba(1,:)'), [amat.*ba(2:end,:)', -alpha], [], 'implicit wing profile drag model');
end