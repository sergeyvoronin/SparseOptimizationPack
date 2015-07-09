function [xsol,num_iters_taken,tElapsed,funct_vals,percent_errors]=conj_grad(A,b,x,tau,x0,maxiters);

tic;

funct_vals = zeros(maxiters,1);
percent_errors = zeros(maxiters,1);

m = size(A,1); n = size(A,2);
At = A';
AtA = At*A;

xn = x0;
sigma = 1e-3;

g_curr=gradF(A,At,b,tau,sigma,xn);

d=-g_curr;
r0 = -g_curr;
rk = r0;

reset_cnt = 0;
numvars = length(xn);

for k = 1:maxiters,

  xcurr=xn;
  fprintf('line searching cong grad testing 2..\n');
  alpha = approximate_line_search(A,At,AtA,b,tau,sigma,xcurr,d);
  %alpha = backtracking_line_search2(A,At,AtA,b,tau,sigma,xcurr,d);
  fprintf('done with alpha = %f\n', alpha);

  xn = xcurr+alpha*d;

  g_old=g_curr;
  g_curr=gradF(A,At,b,tau,sigma,xn);

  rk1 = rk;
  rk = -g_curr;

  sigma = sigma/1.2;

  reset_cnt = reset_cnt + 1;
  if reset_cnt == numvars
    d=-g_curr;
    reset_cnt = 0;
  else
    beta = max((g_curr'*(g_curr-g_old))/(g_old'*g_old),0);
    d=-g_curr+beta*d;
  end

    %xn = hardThreshold(xn,tau);
    %xn = softThreshold(xn,tau);

    if mod(k,10) == 0
        v = A'*(b - A*xn);
        for j=1:length(xn)
            if abs(v(j)) <= tau
                xn(j) = 0;
            end
        end
    end

    percent_errors(k) = 100*norm(xn - x)/norm(x);
    funct_vals(k) = norm(A*xn - b,2)^2 + 2*tau*norm(xn,1);
    fprintf('percent_errors(%d) = %f\n', k, percent_errors(k));
    fprintf('funct_vals(%d) = %f\n', k, funct_vals(k));

    fprintf('norm(r0) = %f\n', norm(r0));
    fprintf('norm(rk1) = %f\n', norm(rk1));
    fprintf('norm(rk) = %f\n', norm(rk));
    
    if k>40 && (norm(rk1) < norm(rk))
        fprintf('Terminating loop due to norm(rk) condition\n');
        xsol = xn;
        num_iters_taken = k;
        tElapsed = toc();
        return;
    end

    eps = 0.05;
    if k>40 && (norm(rk) < eps*norm(r0))
        fprintf('Terminating loop due to eps condition\n');
        xsol = xn;
        num_iters_taken = k;
        tElapsed = toc();
        return;
    end


  if (k == maxiters)
    fprintf('Terminating loop\n');
    fprintf('norm(r0) = %f\n', norm(r0));
    fprintf('norm(rk) = %f\n', norm(rk));
  end %if
end %for

xsol=xn;
num_iters_taken = k;

% record time
tElapsed = toc();

