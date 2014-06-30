function [y, dydx, dydalpha] = lse_scaled(x, alpha)
%log sum exp function with derivatives
%sums across the second dimension of x; 
%returns one y for every row of x
%dydx gives the deriv of each y wrt each x 
%alpha is softness parameter \in R

%note that lse_scaled is a mapping R^n --> R, where n==size(x,2)

[~, n] = size(x);

m = max(x, [], 2);  %maximal x values
h = x - repmat(m, 1, n);   %distance from m; note h <= 0 for all entries
%(unless x has infs; in this case h has nans; can prob deal w/ this gracefully)
%should also deal with alpha==inf case gracefully

expo = exp(alpha*h);
sumexpo = sum(expo, 2);

L = log(sumexpo)/alpha;
y = L + m;

if nargout > 1
    dydx = expo./repmat(sumexpo, 1, n);
    %note that sum(dydx,2)==1, i.e. dydx is a probability distribution
    dydalpha = (sum(h.*expo, 2)./sumexpo - L)/alpha;
end

end