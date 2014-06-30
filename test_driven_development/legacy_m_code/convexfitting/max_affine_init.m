function ba = max_affine_init(x, y, K, varargin)
%initializes max-affine fit to data (y, x)
%ensures that initialization has at least K+1 points per partition (i.e.
%per affine function)

defaults.bverbose = true;
options = process_options(defaults, varargin{:});

npt = size(x,1);
dimx = size(x,2);
X = [ones(npt, 1), x];
ba = zeros(dimx+1,K);
if K*(dimx+1) > npt
    error('Not enough data points');
end

%choose K unique indices
randinds = randperm(npt, K);

%partition based on distances
sqdists = zeros(npt, K);
for k = 1:K
    sqdists(:,k) = sum((x - repmat(x(randinds(k),:), npt, 1)).^2, 2);
end
[~, mindistind] = min(sqdists, [], 2);  %index to closest k for each data pt

%loop through each partition, making local fits
%note we expand partitions that result in singular least squares problems
%why this way? some points will be shared by multiple partitions, but
%resulting max-affine fit will tend to be good. (as opposed to solving least-norm version)
for k = 1:k
    inds = mindistind == k;
    
    %before fitting, check rank and increase partition size if necessary
    %(this does create overlaps)
    if rank(X(inds, :)) < dimx + 1
        [~, sortdistind] = sort(sqdists(:,k));
        i = sum(inds);  %i is number of points in partition
        iinit = i;
        if i < dimx+1
            %obviously, at least need dimx+1 points. fill these in before
            %checking any ranks
            inds(sortdistind(i+1:dimx+1)) = 1;
            i = dimx+1;
        end
        %now add points until rank condition satisfied
        while rank(X(inds, :)) < dimx+1
            i = i+1;
            inds(sortdistind(i)) = 1;
        end
        
        if options.bverbose
            disp(['max_affine_init: Added ', num2str(i-iinit), ' points to partition ', num2str(k), ' to maintain full rank for local fitting.']);
        end
        
    end
    
    %now create the local fit
    ba(:,k) = X(inds, :)\y(inds);
    
end

end