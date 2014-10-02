function [y, dydx, dydalpha] = lse_implicit(x, alpha)
%implicit log sum exp function with derivatives
%sums across the second dimension of x;
%alpha should be a rox vector with length(alpha)==size(x,2)
%returns one y for every row of x
%dydx gives deriv of each y wrt each x
%alpha is local softness parameter for each column of x

%lse_implicit is a mapping R^n --> R, where n==size(x,2)
%implementation: newton raphson steps to find f(x,y) = 0

tol = 10*eps;
[npt, nx] = size(x);

if all(size(alpha) == [nx, 1])
    alpha = alpha';
end
if ~all(size(alpha) == [1, nx])
    error('alpha size mismatch');
end
alphamat = repmat(alpha, npt, 1);

m = max(x, [], 2);  %maximal x values
h = x - repmat(m, 1, nx);   %distance from m; note h <= 0 for all entries
%(unless x has infs; in this case h has nans; can prob deal w/ this gracefully)
%should also deal with alpha==inf case gracefully

L = zeros(npt, 1);  %initial guess. note y = m + L
Lmat = repmat(L, 1, nx);

%initial eval
expo = exp(alphamat.*(h-Lmat));
alphaexpo = alphamat.*expo;
sumexpo = sum(expo, 2);
sumalphaexpo = sum(alphaexpo, 2);
f = log(sumexpo);
dfdL = -sum(alphaexpo, 2)./sumexpo;
neval = 1;
i = abs(f) > tol;   %inds to update
%disp(['max newton-raphson residual: ', num2str(max(abs(f)))]);

while any(i)
    L(i) = L(i) - f(i)./dfdL(i);    %newton step
    
    %re-evaluate
    Lmat(i,:) = repmat(L(i), 1, nx);
    expo(i,:) = exp(alphamat(i,:).*(h(i,:)-Lmat(i,:)));
    alphaexpo(i,:) = alphamat(i,:).*expo(i,:);
    sumexpo(i) = sum(expo(i,:), 2);
    sumalphaexpo(i,:) = sum(alphaexpo(i,:), 2);
    f(i) = log(sumexpo(i));
    dfdL(i) = -sumalphaexpo(i,:)./sumexpo(i);
    neval = neval + 1;
    if neval > 40
        disp('');
    end
    
    %update inds that need to be evaluated
    i(i) = abs(f(i)) > tol;
    %disp(['max newton-raphson residual: ', num2str(max(abs(f)))]);
end

disp(['lse_implicit converged in ', num2str(neval), ' newton-raphson steps']);

y = m + L;

if nargout > 1
    dydx = alphaexpo./repmat(sumalphaexpo, 1, nx);
    dydalpha = (h - Lmat).*expo./repmat(sumalphaexpo, 1, nx);
end

