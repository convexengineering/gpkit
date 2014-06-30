function [params, RMStraj] = LMTR(residfun, initparams, varargin)

%locally solve nonlinear least squares problem
%using trust region algorithm motivated by Levenberg-Marquardt

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
defaults.maxiter = 80;
defaults.maxtime = inf;
defaults.initdelta = 1;
defaults.rhomin = 0.1;
defaults.deltamax = Inf;
options = process_options(defaults, varargin{:});

if options.bverbose
    %define display formatting if required
    formatstr1 = ' %5.0f   %9.6g    %9.3g\n';
    formatstr =  ' %5.0f   %9.6g    %9.3g    %8.4g %12.4g  %12.4g   %8.4g   %8.4g\n';
end

%get residual values and size at initial point; extract size info
params = initparams;
[r, J] = residfun(params);
[npt, sb1] = size(r);
if sb1 > 1
    error('residfun should return a column vector');
elseif any(size(J) ~= [npt, nparam])
    error('Jacobian size inconsistent');
end

%"accept" initial point
rr = r'*r;
rms = sqrt(rr/npt);
point_accepted = true;  %accept initial point so model gets updated
maxgrad = 2*norm(J'*r, inf);

%initializations
iter = 1;   %number of function eval trials
naccept = 0;  %number of accepted steps
RMStraj = zeros(options.maxiter, 1);
RMStraj(1) = rms;
delta = options.initdelta;

%trust region subproblem: initialize terms that don't change iter to iter
prob.qcsubk = ones(nparam,1);
prob.qcsubi = (1:nparam)';
prob.qcsubj = (1:nparam)';

if options.bverbose
    %display info for 1st iter
    fprintf( ...
        ['\n                      First-Order                            Norm of \n', ...
        '   Iter    Residual    optimality     rho        Delta        step     Jwarp        alpha\n']);
    fprintf(formatstr1,iter,rms,maxgrad);
end

while true
    %test for exit cases
    if iter == options.maxiter
        if options.bverbose, disp('Reached maxiter'); end
        break;
    elseif toc > options.maxtime
        if options.bverbose, disp(['Reached maxtime (', num2str(options.maxtime), ' seconds)']); end
        break;
    end
    
    %if point accepted, update model, check gradient for exit case
    if(point_accepted)
        %model is that r = r0+J*step, so rr ~ rr0 + step'*JJ*step + 2J'*step
        %we actually optimize 1/2*step'JJstep + J'*step 
        %mosek includes 1/2 in quadratic terms automatically
        % optimal model objective *2 is predicted decrease in r'r
        prob.c = J'*r;  % == -g, the gradient of objective wrt delta
        %check gradient for exit case here
        JJ = J'*J;
        D = diag(JJ);
        
        %set up trust region subproblem
        %form
        [prob.qosubi, prob.qosubj, prob.qoval] = find(sparse(tril(JJ)));
        prob.a = sparse(1,size(JJ,2));  %no linear terms in constraints
        %can add bound constraints on step *and* params using blx, bux
        
        %elliptical step bound (2-norm)
        prob.qcval = D; 
        
    end
    
    iter = iter + 1;
    
    %solve trust region subproblem as a SOCP
    prob.buc = 1/2*delta;   %mosek assumes 1/2 in objective, so add here too
    [return_code, res] = mosekopt('minimize echo(0)', prob);
    if return_code ~= 0
        error(['mosek returned return code ', res.rcodestr]);
    end
    
    
    step = res.sol.itr.xx;
    trialp = params + step;
    [trialr, trialJ] = residfun(trialp);
    trialrr = trialr'*trialr;
    rho = (rr - trialrr)/(-2*res.sol.itr.pobjval);
    RMStraj(iter) = sqrt(trialrr/npt);
    
    %next bit of logic follows NumericalOptimization algorithm 4.1
    if rho > options.rhomin     %also represented by 'eta'
        %accept step
        params = trialp;
        J = trialJ;
        r = trialr;
        rr = trialrr;
        rms = sqrt(rr/npt);
        maxgrad = 2*norm(J'*r, inf);
        point_accepted = true;
    else
        point_accepted = false;
    end
    
    %display info
    dsp();
        
    if rho < .25    %actual decrease too small; shrink trust region
        delta = delta/4;
    else %step decreased residual appreciably
        if rho > .75 && abs(sum(D.*step.^2)/delta - 1) < 1e-6
            %if model in good agreement and trust region constrained step
            delta = min(2*delta, options.deltamax);
            %else leave delta alone
        end
    end
        
    
    %be sure to update point_accepted based on trial
    
end

RMStraj(iter+1:end) = [];


    function dsp()
        %display current progress
        %note that dsp should be called immediately after evaluating a
        %trial point (and before updating delta) in order for delta to
        %correspond to the trust region used in the current iterate
        if(options.bverbose)
            fprintf(formatstr,iter,RMStraj(iter),maxgrad,rho,delta,norm(step),cond(full(JJ)), 1/params(end));
            %str = ['i=', num2str(iter), ' | bestrms=', num2str(rms), ' | lambda=', num2str(lambda), ' | maxg=', num2str(norm(r'*J, 'inf'))];
            %disp(str);
        end
    end


end