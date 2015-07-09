% we implement irls iterative solve for regulrizing Ax=b
% via minimization of J(x) = 2*tau*norm(xn,1) + norm(A*xn-b,2)^2;
% we have A,x,b; norm(A)<1
% see svoronin thesis
function [x_sol,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = irls_solve_iterative(A,b,x,tau,x0,TOL,maxiters) 

tStart = tic;
m = size(A,1);
n = size(A,2);
quit_loop = 0;

xn = x0;
wns = ones(n,1);
en = 1;
gamma = 0.1;

% calculate z = A^t*b
z = A'*b;

fprintf('in irls_solve_iterative, starting loop..\n');
fprintf('maxiters = %d, tau = %f\n', maxiters, tau);
for i=1:maxiters
    if quit_loop ~= 1
            fprintf('doing irls iterative iter %d of %d with tau = %f\n', i, maxiters, tau);

            num_iters_taken = i;

            % compute residual
            residuals(i) = norm(A*xn - b,2)^2;
            Js(i) = 2*tau*norm(xn,1) + norm(A*xn-b,2)^2;
            xnx_errors(i) = 100*norm(xn-x)/norm(x);

            % record old solution
            xn1 = xn;

            % compute weights
            for ind=1:length(wns)
                %wns(ind) = 1./max(abs(xn(ind)),en);
                wns(ind) = 1./(1 + tau/sqrt(xn(ind)^2 + en^2));
            end

            % compute new solution component wise 
            AtAxn = A'*(A*xn);
            for ind=1:length(xn)
                xn(ind) = wns(ind) * ( xn(ind) + z(ind) - AtAxn(ind) );
            end

            % update en
            en = min( en, sqrt(norm(xn - xn1)) + gamma^n );

            
            % compute diff between un and un1 and quit if small
            if(i>3 && (100*(norm(xn - xn1)/norm(xn)))<TOL)
                fprintf('reached convergence..\n');
                quit_loop = 1;
            end
    else
        break;
    end
end
fprintf('done with irls_solve_iterative.. num_iters_taken = %d\n', num_iters_taken); 

xnx_errors = xnx_errors(1:num_iters_taken);

% record solution
x_sol = xn;

% record time
tElapsed = toc(tStart);

