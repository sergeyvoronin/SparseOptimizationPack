function [xsol,num_iters_taken,tElapsed,funct_vals,percent_errors] = newtons_method_solve(A,b,x,tau,x0,maxiters) 

tic;
m = size(A,1);
n = size(A,2);
At = A';
AtA = At*A;

funct_vals = zeros(maxiters,1);
percent_errors = zeros(maxiters,1);

num_iters_taken = 0;
quit_loop = 0;

% do three iterations of fista to warm start newton!
fprintf('warm start newton..\n');
TOL = 0;
[xn,num_iters_taken,tElapsed]=conj_grad(A,b,x,tau,x0,15);
fprintf('end warm start newton.\n');
maxiters = 15;

mu = 1.5;
sigma = 1e-3;

fprintf('in newtons method solve, starting loop..\n');
fprintf('maxiters = %d, tau = %f\n', maxiters, tau);
for i=1:maxiters
    if quit_loop ~= 1
        fprintf('doing newton iter %d of %d with tau = %f\n', i, maxiters, tau);

        num_iters_taken = i;

        hessian_matx = hessianF(A,At,AtA,b,tau,sigma,xn);
        rhs = -gradF(A,At,b,tau,sigma,xn);

        %delta_x = hessian_matx \ rhs; 
        delta_x = pcg(hessian_matx'*hessian_matx, hessian_matx'*rhs, 0.001, 15); 

        xn1 = xn;
        xn = xn + delta_x;
        

        % remove small components
        xn = softThreshold(xn,tau); 

        sigma = sigma/1.5;

        % compute diff between un and un1 and quit if small
        %if(i>3 && (100*(norm(xn - xn1)/norm(xn)))<TOL)
        %    fprintf('reached convergence..\n');
        %    quit_loop = 1;
        %end
    else
        break;
    end
end

% record solution
xsol = xn;

% record time
tElapsed = toc();

end
