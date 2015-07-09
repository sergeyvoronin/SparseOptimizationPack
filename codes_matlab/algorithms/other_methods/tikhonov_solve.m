% we implement tikhonov solve for regularizing Ax=b
% via minimization of J(x) = 2*tau*norm(xn,2)^2 + norm(A*xn-b,2)^2;
% we have A,x,b; norm(A)<1
% we solve by pre-conditioned conjugate gradients:
% (A'*A + \tau*I) x_sol = A'*b
% svoronin
function [x_sol,num_iters_taken,tElapsed] = tikhonov_solve(A,b,x,tau,x0,TOL,maxiters) 

tStart = tic;
m = size(A,1);
n = size(A,2);

% record solution via bicg A is mxn, A' is nxm so A'*A is nxm by mxn = nxn
AtA = A'*A;
[x_sol,flag,relres,num_iters_taken] = bicg((AtA + tau*eye(n,n)),A'*b,TOL,maxiters,[],[],x0);

% record time
tElapsed = toc(tStart);

