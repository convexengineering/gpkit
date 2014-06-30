function matout = repcols(matin, n)
%replicate columns of a matrix
%returns a matrix with n times as many columns as matin
%example: if matin is [a b c], repcols(matin, 2) returns [a a b b c c].
[nrow, ncol] = size(matin);

matout = reshape(repmat(matin, n, 1), nrow, ncol*n);

end