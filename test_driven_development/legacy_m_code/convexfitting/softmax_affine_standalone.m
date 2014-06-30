function [y, dydp] = softmax_affine(x,params)
%params may come in as
%  1) a cell {ba, alpha}, or 
%  2) a vector (with alpha last)
%after reshaping (column-major), ba is dimx+1 by K
%first row is b
%rest is a

if iscell(params)
    ba = params{1};
    alpha = params{2};
else
    ba = params(1:end-1);
    alpha = params(end);
end

%check sizes
[npt, dimx] = size(x);
K = numel(ba)/(dimx+1);
ba = reshape(ba,dimx+1,K);

%augment data with column of ones
X = [ones(npt,1), x];

%compute affine functions
z = X*ba;
m = max(z,[],2);
delta = z - repmat(m, 1, K);

%function value
expo = exp(alpha*delta);
sumexpo = sum(expo,2);
lse = log(sum(expo,2))/alpha;
y = m + lse;

if(nargout > 1)
	%todo deal with sparse case
	%todo deal with iscell(params) case
	%for each K, ba is dimx+1
	dydba = repcols(expo./repmat(sumexpo, 1, K), dimx+1) .* repmat(X, 1, K);
    %still need to deal with alpha
    dydalpha = (sum(expo.*delta,2)./sumexpo - lse)./alpha;
    dydp = [dydba, dydalpha];
end


