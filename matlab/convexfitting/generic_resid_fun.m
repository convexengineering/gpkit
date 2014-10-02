function [r, drdp] = generic_resid_fun(yfun, xdata, ydata, params)
    %generic residual function -- converts yfun(xdata,params)
    %to a residfun [r, drdp] = generic_resid_fun(yfun, xdata, ydata, params)
    %used by nonlinear least squares fitting algorithms
    %to get a residual function [r,drdp] = residfun(params),
    %use rfun = @(p) generic_resid_fun(@yfun, xdata, ydata, p)
    
    %note this function defines resids as + when yhat > ydata 
    %(opposite of typical conventions, but eliminates need for sign change)

    [yhat, drdp] = yfun(xdata, params);
    r = yhat - ydata;
    
end