% cvx_begin sdp quiet
% variable t
% variable x(4,1)
% minimize t
% subject to
% [speye(10), A*x-b; (A*x-b)' t] > 0
% [speye(4), x; x', 1] > 0
% cvx_end

% cvx_begin quiet
% variable x(4,1)
% minimize norm(A*x-b)
% subject to 
% norm(x, 2) < 1
% cvx_end

clear prob;

AA = A'*A;
prob.c = -A'*b;
[prob.qosubi, prob.qosubj, prob.qoval] = find(sparse(tril(AA)));
prob.a = sparse(1,size(AA,2));
prob.buc = 1/2;

% prob.cones = cell(1,1);
% prob.cones{1}.type = 'MSK_CT_QUAD';
% prob.cones{1}.sub = [size(AA,2)+1, 1:size(AA,2)];

%quadratic constraints on x
prob.qcsubk = ones(size(AA,2),1);
prob.qcsubi = (1:size(AA,2))';
prob.qcsubj = (1:size(AA,2))';
prob.qcval = diag(AA);

[r, res] = mosekopt('minimize echo(0)', prob);