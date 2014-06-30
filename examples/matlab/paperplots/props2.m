%generate 1000N thrust
V = 55;
ep = 0.05;
A = 2.4;
T = 1000;
rho = 1.23;

cvx_begin gp

variables Wt Wa vt va eta lambda

maximize eta

subject to
Wt*vt == Wa*va
Wa > V + va
V/lambda > Wt + vt
Wt > T/(2*rho*A*vt) + ep*Wa
Wt > eta/lambda*Wa + ep*Wt*eta/lambda + ep*Wa

%note: why not make prop area ridiculously large?
%this would be very lightly loaded (low T/A), 
%which would make epsilon (cd/cl) get large
%this effect depends on section chords, which we still need to model

cvx_end