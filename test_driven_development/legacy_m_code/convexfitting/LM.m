function [params, RMStraj] = LM(residfun, initparams, varargin)

%Levenberg-Marquardt algorithm
%minimizes sum of squared error of residfun(params)
%INPUTS
%residfun should have the form [r, drdp] = residfun(params)
%for curve fitting,
% if residfun is (ydata - y(params)), drdp is -dydp
% if residfun is (y(params) - ydata), drdp is dydp
%params is a column vector of initial param guesses
%varargin is a list of option, value pairs
%OUTPUTS
%params: best params found
%RMStraj: history of RMS errors after each step (first point is initialization)

tic;
%check incoming params
[nparam, sb1] = size(initparams);
if sb1 > 1
    error('params should be a column vector');
end

%set defaults; incorporate incoming options
defaults.bverbose = true;
%defaults.bplot = true;
defaults.lambdainit = 0.02;
defaults.maxiter = 200;
defaults.maxtime = 5;
defaults.tolgrad = sqrt(eps);
defaults.tolrms = 1e-7;
options = process_options(defaults, varargin{:});

%define display formatting if required
formatstr1 = ' %5.0f   %9.6g    %9.3g\n';
formatstr =  ' %5.0f   %9.6g    %9.3g %12.4g  %12.4g   %8.4g\n';

%get residual values and jacobian at initial point; extract size info
params = initparams;
params_updated = true;
[r, J] = residfun(params);
[npt, sb1] = size(r);
if sb1 > 1
    error('residfun should return a column vector');
elseif any(size(J) ~= [npt, nparam])
    error('Jacobian size inconsistent');
end

%"accept" initial point
rms = norm(r, 2)/sqrt(npt);
maxgrad = norm(r'*J, Inf);
prev_trial_accepted = false;

%initializations
iter = 1;
Jissparse = issparse(J);
diagJJ = sum(J.*J, 1)';
zeropad = zeros(nparam,1);
lambda = options.lambdainit;
RMStraj = zeros(options.maxiter,1);
RMStraj(1) = rms;
gradcutoff = options.tolgrad;  %*max(maxgrad, 1); 

%display info for 1st iter
if(options.bverbose)
fprintf( ...
    ['\n                      First-Order                   Norm of \n', ...
        '   Iter    Residual    optimality      Lambda        step     Jwarp\n']);
        fprintf(formatstr1,iter,rms,maxgrad);
end

%main loop
while true
    
    %test for exit cases
    if iter == options.maxiter
        if options.bverbose, disp('Reached maxiter'); end
        break;
    elseif toc > options.maxtime
        if options.bverbose, disp(['Reached maxtime (', num2str(options.maxtime), ' seconds)']); end
        break;
    elseif iter > 2 && abs(RMStraj(iter)-RMStraj(iter-2)) < RMStraj(iter)*options.tolrms
        %should really only allow this exit case if trust region
        %constraint is slack
        if options.bverbose, disp(['RMS changed less than tolrms']); end
        break;
    end
    
    iter = iter + 1;
    
    %compute diagonal scaling matrix based on current lambda and J
    %note this matrix changes every iteration, since either lambda or J
    %(or both) change on every iteration
    if Jissparse
        D = spdiags(sqrt(lambda*diagJJ), 0, nparam, nparam);
    else
        D = diag(sqrt(lambda*diagJJ));
    end
    
    %update augmented least squares system
    if params_updated
        %J = trialJ;
        diagJJ = sum(J.*J, 1)';
        augJ = [J; D];
        augr = [-r; zeropad];
    else
        augJ(npt+1:end, :) = D;
    end
    
    %compute step for this lambda
    step = augJ\augr;
    trialp = params + step;
    
    %check function value at trialp
    [trialr, trialJ] = residfun(trialp);
    trialrms = norm(trialr)/sqrt(npt);
    RMStraj(iter) = trialrms;
    
    %accept or reject trial params
    if trialrms < rms
        params = trialp;
        J = trialJ;
        r = trialr;
        rms = trialrms;
        maxgrad = norm(r'*J, Inf);
        dsp();  %dsp here so that all grad info is for updated point, but lambda not yet updated
        if maxgrad < gradcutoff
            if options.bverbose, disp('1st order optimality attained'); end
            break;
        end
        if prev_trial_accepted && iter > 1
            lambda = lambda/10;
        end
        prev_trial_accepted = true;
        params_updated = true;
    else
        dsp();
        lambda = lambda*10;
        prev_trial_accepted = false;
        params_updated = false;
    end
    
end

RMStraj(iter+1:end) = [];
disp(['Final RMS: ', num2str(rms)]);
% if options.bplot
%     fs = 14;
%     figure(1); clf(1);
%     semilogy(RMStraj);
%     xlabel('iteration', 'fontsize', fs);
%     ylabel('RMS error', 'fontsize', fs);
% end

    function dsp()
        %display current progress
        %note that dsp should be called immediately after evaluating a
        %trial point (and before updating lambda) in order for lambda to
        %correspond to the trust region used in the current iterate
        if(options.bverbose)
            fprintf(formatstr,iter,trialrms,maxgrad,lambda,norm(step),full(max(diagJJ)/min(diagJJ)));
            %str = ['i=', num2str(iter), ' | bestrms=', num2str(rms), ' | lambda=', num2str(lambda), ' | maxg=', num2str(norm(r'*J, 'inf'))];
            %disp(str);
        end
    end

end
