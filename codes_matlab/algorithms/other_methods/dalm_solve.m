% dual space dalm fast solve 
% based on: http://www.eecs.berkeley.edu/~yang/software/l1benchmark/
% this is the fast version of the algorithm from the toolbox
function [x_sol,num_iters_taken,tElapsed] = dalm_solve(A,b,x,tau,x0,TOL,maxiters) 

tStart = tic;
m = size(A,1);
n = size(A,2);

num_iters_taken = 0;
quit_loop = 0;

beta = norm(b,1)/m;
betaInv = 1/beta ;

y = zeros(m,1);
xn = x0;    
z = zeros(m+n,1);

temp = A' * y;
f = norm(xn,1);



fprintf('in dalm_fast_solve, starting loop..\n');
fprintf('maxiters = %d, tau = %f\n', maxiters, tau);
for i=1:maxiters
    if quit_loop ~= 1
            fprintf('doing DALM fast iter %d of %d with tau = %f\n', i, maxiters, tau);

            %x_{n+1}=T(x_n+\frac{t_{n}-1}{t_{n+1}} (x^{(n)}-x^{(n-1)}));
            num_iters_taken = i;

            xn1 = xn;
    
            %update z
            temp1 = temp + xn * betaInv;
            z = sign(temp1) .* min(1,abs(temp1));
    
            %compute A' * y    
            g = tau * y - b + A * (beta * (temp - z) + xn);
            %alpha = g' * g / (g' * G * g);
            Ag = A' * g;
            alpha = g' * g / (tau * g' * g + beta * Ag' * Ag);
            y = y - alpha * g;
            temp = A' * y;
            
            %update x
            xn = xn - beta * (z - temp);

            % apply some thresholding scheme
            % below is based on \ell_1 opt conditions
            if mod(i,5) == 0
                v = A'*(b - A*xn);
                for j=1:length(xn)
                    if abs(v(j)) <= tau
                        xn(j) = 0;
                    end
                end
            end

%            absxn = abs(xn);
%            absxn_sorted = sort(absxn,'descend');
%            top33_threshold = absxn_sorted(round(1.5*nnz(x))); 
%            for k=1:length(xn)
%                if abs(xn(k)) < top33_threshold
%                    xn(k) = 0;
%                end
%            end

            if(i>3 && (100*(norm(xn - xn1)/norm(xn)))<TOL)
                fprintf('reached convergence..\n');
                quit_loop = 1;
            end
    else
        break;
    end
end

% record solution
x_sol = xn;

% record time
tElapsed = toc(tStart);

