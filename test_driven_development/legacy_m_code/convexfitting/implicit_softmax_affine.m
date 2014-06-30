function [y, dydp] = implicit_softmax_affine(x, params)
%%params may come in as
%  1) a cell {ba, alpha}, or 
%  2) a vector (with alpha last)
%after reshaping (column-major), ba is dimx+1 by K
%first row is b
%rest is a

[npt, dimx] = size(x);

if iscell(params)
    ba = params{1};
    alpha = params{2};
    K = numel(ba)/(dimx+1);
else
    K = numel(params)/(dimx+2);
    ba = params(1:end-K);
    alpha = params(end-K+1:end);
end

%reshape ba to matrix
ba = reshape(ba, dimx+1, K);

if any(alpha <= 0)
    y = Inf*ones(npt,1);
    dydp = nan;
    return;
end

%augment data with column of ones
X = [ones(npt,1), x];

%compute affine functions
z = X*ba;

if nargout == 1
    y = lse_implicit(z, alpha);
else
    [y, dydz, dydalpha] = lse_implicit(z, alpha);
    dydba = repcols(dydz, dimx+1) .* repmat(X, 1, K);
    dydp = [dydba, dydalpha];
end

end



