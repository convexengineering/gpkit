function [ba, residtraj] = magnani_boyd(x, y, K, varargin)

%carries out the Magnani-Boyd Least Squares Partition Algorithm to fit a
%convex max-affine model to data (x, y)

defaults.maxiter = 100;
defaults.relcutoff = 1e-6; %stop when abs(resid-oldresid)/oldresid < relcutoff
defaults.bverbose = true;
options = process_options(defaults, varargin{:});

npt = size(x, 1);
dimx = size(x,2);
X = [ones(npt,1), x];
residfun = @(yhat) norm(yhat - y, 2)/sqrt(npt);

iter = 0;
%oldresid = Inf; %used to keep track of changes in resid
residtraj = zeros(options.maxiter, 1);
while(iter < options.maxiter)
    iter = iter + 1;
    
    %start by updating ba (or initializing it if first iter)
    if(iter == 1)
        ba = max_affine_init(x, y, K);
    else
        %fit each local partition with an affine plane
        for k = 1:K
            inds = partition==k;
            if rank( X(inds, :) ) >= dimx+1
                %only do the update for partitions whose rank is large
                %enough
                ba(:, k) = X(inds, :)\y(inds);  %MATLAB warning says size changes, but this is false.
            end
        end
    end

    %calculate yhat and new partitioning
    [yhat, partition] = max(X*ba, [], 2);
    residtraj(iter) = residfun(yhat);
    
    if options.bverbose
        str = ['i=', num2str(iter), ': resid=', num2str(residtraj(iter))];
        disp(str);
    end
    
    if iter > 1 && abs(residtraj(iter)-residtraj(iter-1))/residtraj(iter) < options.relcutoff
        if options.bverbose
            disp('magnani_boyd: exiting on relative change in resid');
        end
        break;
    end
    
end

%update residtraj to cut off unused entries at end
residtraj(iter+1:end) = [];

end