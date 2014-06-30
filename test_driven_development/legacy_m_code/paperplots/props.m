lambda = 0.2;

cvx_begin gp

variables z eta CT
maximize eta
subject to

CT/lambda^2 > 50
z^2 > 1 + CT/lambda^2
2 > eta + eta*z

cvx_end