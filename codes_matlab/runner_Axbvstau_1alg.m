% here we construct curve ||Ax - b|| vs tau with one algorithm
function runner_Axbvstau_1alg(system_data_str, vars_filename_str, alg_name, num_taus, maxiters, TOL, plot_debug);  

% load system data with A,b,x, etc 
load(system_data_str);

% set range of taus
taus = zeros(num_taus,1);
tau_max = full(max(A'*b))/1.5;
tau_min = full(max(A'*b))/5e8;
log_tau_max = log(tau_max);
log_tau_min = log(tau_min);
log_tau_step = (log_tau_max-log_tau_min)/(num_taus-1);
for i=1:num_taus
    log_tau = log_tau_max - log_tau_step*(i-1);
    taus(i) = exp(log_tau);
end
tau_fracs = taus/full(max(A'*b));

% set up structures
final_residuals_alg = zeros(num_taus,1);
num_iters_alg = zeros(num_taus,1);
percent_errors_alg = zeros(num_taus,1);
num_nnzs_alg = zeros(num_taus,1);
num_nnzs_x = zeros(num_taus,1); % this is the same for each tau but we should average it
total_runtimes_alg = zeros(num_taus,1);
percent_error_min_alg = 1000;
best_tau_alg = 1;


% run the algorithm at these taus
for tau_num=1:num_taus
    tau = taus(tau_num);
    fprintf('processing tau %d of %d = %f..\n', tau_num, num_taus, tau);

    % run alg at this tau
    % reuse the solution at the previous step
    if tau_num==1
        x0 = zeros(n,1);
    else
        x0 = x_sol_alg;
    end

    if strcmp(alg_name,'fista') == 1
        [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_fista_solve(A,b,x,tau,x0,TOL,maxiters);
    elseif strcmp(alg_name,'tikhonov') == 1
        [x_sol_alg,num_iters_taken,tElapsed] = tikhonov_solve(A,b,x,tau,x0,TOL,maxiters); 
    elseif strcmp(alg_name,'fivta') == 1
        %K0 = sparsityX*n*10; % for blocked input 
        K0 = sparsityX*n;
        [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_fivta_solve(A,b,x,tau,x0,K0,TOL,maxiters); 
    elseif strcmp(alg_name,'irls_iterative') == 1
        [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = irls_solve_iterative(A,b,x,tau,x0,TOL,maxiters);
    else
        fprintf('unrecognized alg name %s\n', alg_name);
        return;
    end

    final_residuals_alg(tau_num) = norm(A*x_sol_alg - b,2);
    num_iters_alg(tau_num) = num_iters_taken;
    percent_errors_alg(tau_num) = 100*norm(x_sol_alg - x)/norm(x);
    num_nnzs_alg(tau_num) = nnz(x_sol_alg);
    num_nnzs_x(tau_num) = nnz(x);
    total_runtimes_alg(tau_num) = tElapsed;

    if percent_errors_alg(tau_num) < percent_error_min_alg
        best_tau_alg = tau;
        percent_error_min_alg = percent_errors_alg(tau_num);
    end
end


% rerun at tau that best matches the noise level  
% find intersection point
intersection_point = 1;
noise_val = norm(noise,2);
for i=2:length(final_residuals_alg)
    if final_residuals_alg(i) < noise_val && final_residuals_alg(i-1) >= noise_val
        intersection_point = i-1;
    end
end
tau_noise_match = taus(intersection_point);

maxiters_noise_match = 2*maxiters;
tau = tau_noise_match;
fprintf('rerun alg at noise match tau = %f\n', tau);
if strcmp(alg_name,'fista') == 1
    [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_fista_solve(A,b,x,tau,x0,TOL,maxiters_noise_match);
elseif strcmp(alg_name,'tikhonov') == 1
    [x_sol_alg,num_iters_taken,tElapsed] = tikhonov_solve(A,b,x,tau,x0,TOL,maxiters_noise_match); 
elseif strcmp(alg_name,'fivta') == 1
    %K0 = sparsityX*n*10;
    K0 = sparsityX*n;
    [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_fivta_solve(A,b,x,tau,x0,K0,TOL,maxiters_noise_match); 
elseif strcmp(alg_name,'irls_iterative') == 1
    [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = irls_solve_iterative(A,b,x,tau,x0,TOL,maxiters_noise_match);
else
    fprintf('unrecognized alg name %s\n', alg_name);
    return;
end

% record 'noise match' solution
x_sol_alg_noise_match = x_sol_alg;


% rerun at best tau - tau that gives the smallest percent error
maxiters_best = 2*maxiters;
tau = best_tau_alg;
fprintf('rerun alg at best tau = %f\n', tau);
if strcmp(alg_name,'fista') == 1
    [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_fista_solve(A,b,x,tau,x0,TOL,maxiters_best);
elseif strcmp(alg_name,'tikhonov') == 1
    [x_sol_alg,num_iters_taken,tElapsed] = tikhonov_solve(A,b,x,tau,x0,TOL,maxiters_best); 
elseif strcmp(alg_name,'fivta') == 1
    K0 = sparsityX*n;
    [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = thr_fivta_solve(A,b,x,tau,x0,K0,TOL,maxiters_best); 
elseif strcmp(alg_name,'irls_iterative') == 1
    [x_sol_alg,num_iters_taken,tElapsed,residuals,Js,xnx_errors] = irls_solve_iterative(A,b,x,tau,x0,TOL,maxiters_best);
else
    fprintf('unrecognized alg name %s\n', alg_name);
    return;
end

% record 'best' solution
x_sol_alg_best = x_sol_alg;

% define stuff for plotting
noise_line = repmat(norm(noise,2),num_taus,1);

% save stuff - recover everything from here when taking stats
save(vars_filename_str);


% if plot_debug is set make some plots and pause for a bit to review
if plot_debug == 1
    figure(1);
    hold on;
    plot(final_residuals_alg,'r');
    %plot(noise_line,'b--');
    %set(gca,'xdir','reverse');
    title('residuals vs taus');
    hold off;

    figure(2);
    hold on;
    plot(num_iters_alg,'r');
    title('number of iterations vs taus');
    %set(gca,'xdir','reverse');
    hold off;

    figure(3);
    hold on;
    plot(percent_errors_alg,'r');
    title('percent errors vs taus');
    %set(gca,'xdir','reverse');
    hold off;

    figure(4)
    plot(tau_fracs,'r');
    title('tau fractions');

    figure(5)
    plot(svdsA,'r');
    title('svds of A');

    % plot best reconstructions (at optimally detected taus)
    figure(6)
    hold on;
    plot(x,'r--*','linewidth',2);
    plot(x_sol_alg_best,'b','linewidth',2);
    legend('original','best alg','best alg2');
    hold off;

    fprintf('\npause to review plots..\n'); 
    pause(10);
end

