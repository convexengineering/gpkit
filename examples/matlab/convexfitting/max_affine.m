function [y, dydba] = max_affine(x,ba)
%ba may come in as a matrix or as a vector
%after reshaping (column-major), ba is dimx+1 by K
%first row is b
%rest is a

[npt, dimx] = size(x);
K = numel(ba)/(dimx+1);
ba = reshape(ba,dimx+1,K);

%augment data with column of ones
X = [ones(npt,1), x];

if(nargout == 1)
    y = max(X*ba, [], 2);
else
    [y, partition] = max(X*ba, [], 2);
    
%   (non-sparse version)
%     dydba = zeros(size(x,2)+1, K, npt);
%     %todo - make this sparse
%     for k = 1:K
%         inds = partition == k;
%         dydba(:,k,inds) = [ones(1,nnz(inds)); x(inds,:)'];
%     end

    dydba = spalloc(npt, (dimx+1)*K, npt*(dimx+1));
    for k = 1:K
        inds = partition==k;
        indadd = (dimx+1)*(k-1);
        dydba(inds, indadd+(1:dimx+1)) = X(inds,:);
    end
end