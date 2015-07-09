% we implement thresholded fivta for regularizing Ax=b
% see paper:
% "A new iterative firm-thresholding algorithm for inverse problems with 
% sparsity constraints"
% svoronin
function [x_sol,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_fivta_solve(A,b,x,tau,x0,K0,TOL,maxiters) 

tStart = tic;
m = size(A,1);
n = size(A,2);

% generate sequence of tn's needed for thresholded FISTA
whos maxiters
fprintf('maxiters = %d\n', maxiters);
tns = repmat(0,maxiters+10,1);
tns(1) = 1;
for i=2:length(tns)
    tns(i) = (1 + sqrt(1 + 4*tns(i-1)^2))/2;
end

% calculate z = A^t*b
z = A'*b;

num_iters_taken = 0;
xn = x0;
xn1 = x0;
quit_loop = 0;
residuals = repmat(0,maxiters,1);
Js = repmat(0,maxiters,1);
xnx_errors = repmat(0,maxiters,1);

K = K0;
L = 0;
%rho = tau/2 * (1 + max(0, (n - L - K)/(n-K)));
rho = tau;

fprintf('in thr_fivta_solve, starting loop..\n');
fprintf('maxiters = %d, tau = %f\n', maxiters, tau);
for i=1:maxiters
    if quit_loop ~= 1
            fprintf('doing thresholded FIVTA iter %d of %d with tau = %f\n', i, maxiters, tau);

            %x_{n+1}=T(x_n+\frac{t_{n}-1}{t_{n+1}} (x^{(n)}-x^{(n-1)}));
            num_iters_taken = i;

            % compute residual
            residuals(i) = norm(A*xn - b,2)^2;
            Js(i) = 2*tau*norm(xn,1) + norm(A*xn-b,2)^2;
            xnx_errors(i) = 100*norm(xn-x)/norm(x);
            
            if i==1
                pn = xn;
            else
                pn = xn + ((tns(i)-1)/tns(i+1))*(xn - xn1);
            end
            
            xn1 = xn; % record x_{n-1}

            vn = A*pn;
            wn = A'*vn;
            [xn,L] = fivtaThreshold(pn + z - wn, rho,tau);


            % recompute rho based on L
            rho = tau/2 * (1 + max(0, (n - L - K)/(n-K)));

            % compute diff between un and un1 and quit if small
            if(i>3 && (100*(norm(xn - xn1)/norm(xn)))<TOL)
                fprintf('reached convergence..\n');
                quit_loop = 1;
            end
    else
        break;
    end
end
fprintf('done with thr_fivta_solve.. num_iters_taken = %d\n', num_iters_taken); 

xnx_errors = xnx_errors(1:num_iters_taken);

% record solution
x_sol = xn;

% record time
tElapsed = toc(tStart);

