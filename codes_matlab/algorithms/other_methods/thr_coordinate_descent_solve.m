% solves l1-min problem via randomized coordinate descent
function [x_sol,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_coordinate_descent_solve(A,b,x,tau,x0,TOL,maxiters,column_norms) 

    A = full(A);
    tStart = tic;
    m = size(A,1);
    n = size(A,2);

    num_iters_taken_cd = 0;
    xn = x0;
    quit_loop = 0;
    residuals = repmat(0,maxiters,1);
    Js = repmat(0,maxiters,1);
    xnx_errors = repmat(0,maxiters,1);

    for i=1:maxiters
        if quit_loop ~= 1

            fprintf('in iteration %d of %d of coordinate descent with tau = %f\n', i, maxiters, tau);

            % compute residual and norms
            num_iters_taken_cd = num_iters_taken_cd + 1;
            residuals(i) = norm(A*xn - b,2)^2;
            Js(i) = 2*tau*norm(xn,1) + norm(A*xn-b,2)^2;
            un1 = 2*tau*norm(xn,1) + norm(A*xn-b,2)^2;
            xnx_errors(i) = 100*norm(xn-x)/norm(x);
            
            % record last x
            xn1 = xn;

            % pick index j at random from [1,n] - over all possible inds, not just the support
            j = round(1 + (n-1)*rand);
            fprintf('picked j = %d ; xn(%d) = %f\n', j, j, xn(j));

            % calculate B_j = sum i=1..m { a_ij ( f_i - sum k=1..n,not j { a_ik x_k } }
%            Bj = 0;
%            for ind=1:m
%                vi = 0;
%                for k=1:n
%                    if k~= j
%                        vi = vi + A(ind,k)*xn(k);
%                    end
%                end
%                Bj = Bj + A(ind,j)*(b(ind) - vi);
%            end
            Bj = 0;
            Axn = A*xn;
            dj = zeros(n,1);
            dj(j) = 1;
            A_j = A*dj; % this is the j-th column
            vi = 0;
            for ind=1:m
                dj2 = zeros(1,m);
                dj2(ind) = 1;
                A_j_ind = dj2*A_j;
                vi = Axn(ind) - A_j_ind*xn(j);
                Bj = Bj + A_j_ind*(b(ind) - vi);
            end

            % soft-threshold Bj at tau
            xj = softThreshold(Bj,tau);

            % update component
            xn(j) = xj/(column_norms(j)^2);
            
            fprintf('reset xn(%d) = %f\n', j, xn(j));

            % compute diff between un and un1 and quit if small
            if(i>3 && (100*(norm(xn - xn1)/norm(xn)))<TOL)
                fprintf('reached convergence..\n');
                quit_loop = 1;
            else
                break;
            end
    end
end

    fprintf('done with thr_coordinate_descent_solve.. num_iters_taken_cd = %d\n', num_iters_taken_cd); 

    % record solution
    x_sol = xn;

    % record total iterations
    num_iters_taken = num_iters_taken_cd;

    % record time
    tElapsed = toc(tStart);
end

