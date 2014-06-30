function [y, dydp] = softmax_affine(x,params,softness_param_is_alpha)
%params may come in as
%  1) a cell {ba, alpha}, or 
%  2) a vector (with alpha last)
%after reshaping (column-major), ba is dimx+1 by K
%first row is b
%rest is a

if nargin == 2
    softness_param_is_alpha = false;    %then it's gamma (gamma = 1/alpha)
end

if iscell(params)
    ba = params{1};
    softness = params{2};
else
    ba = params(1:end-1);
    softness = params(end);
end

if softness_param_is_alpha
    alpha = softness;
else
    alpha = 1/softness;
end

%check sizes
[npt, dimx] = size(x);
K = numel(ba)/(dimx+1);
ba = reshape(ba,dimx+1,K);

if alpha <= 0
    y = Inf*ones(npt,1);
    dydp = nan;
    return;
end

%augment data with column of ones
X = [ones(npt,1), x];

%compute affine functions
z = X*ba;

if nargout == 1
    y = lse_scaled(z, alpha);
else
    [y, dydz, dydsoftness] = lse_scaled(z, alpha);
    if ~softness_param_is_alpha
        dydsoftness = -dydsoftness*(alpha^2);
    end
    dydba = repcols(dydz, dimx+1) .* repmat(X, 1, K);
    dydp = [dydba, dydsoftness];
end

end

