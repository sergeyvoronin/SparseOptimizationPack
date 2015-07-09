% driver script for runner
% this calls runner mutliple times, calculates statistics and saves to filename that is read 
% by plotter to make the plots

function driver_Axbvstau_1alg(alg_name, system_data_dir, vars_filename_str_runner, vars_filename_str_driver, num_taus, num_trials, maxiters, TOL)

% set trial params
plot_debug = 0; % don't make intermediate plots

% set up structures
cellarr_percent_errors_alg = cell(num_trials,1);
cellarr_num_iters_to_converge_alg  = cell(num_trials,1);
cellarr_total_runtimes_alg  = cell(num_trials,1);
cellarr_residuals_alg = cell(num_trials,1);
cellarr_num_nnzs_alg  = cell(num_trials,1);
cellarr_num_nnzs_x  = cell(num_trials,1);

% run and collect stats
for trial_num=1:num_trials
    fprintf('->>> processing %d of %d..\n', trial_num, num_trials);
    pause(2);

    % make string point to data file
    system_data_str = [system_data_dir, '/system', num2str(trial_num), '.mat'];

    runner_Axbvstau_1alg(system_data_str, vars_filename_str_runner, alg_name, num_taus, maxiters, TOL, plot_debug);  


    % load in the variables
    fprintf('loading variables from %s and checking for data\n', vars_filename_str_runner);
    load(vars_filename_str_runner);

    % check for stuff we want
    whos percent_errors_alg num_iters_alg total_runtimes_alg final_residuals_alg num_nnzs_alg num_nnzs_x

    % save to above created structures
    cellarr_percent_errors_alg{trial_num} = percent_errors_alg; 
    cellarr_num_iters_to_converge_alg{trial_num}  = num_iters_alg;
    cellarr_total_runtimes_alg{trial_num}  = total_runtimes_alg;
    cellarr_final_residuals_alg{trial_num} = final_residuals_alg;
    cellarr_num_nnzs_alg{trial_num}  = num_nnzs_alg;
    cellarr_num_nnzs_x{trial_num}  = num_nnzs_x;
end

% compute medians quartiles etc
percent_errors_median = zeros(num_taus,1);
percent_errors_q1 = zeros(num_taus,1);
percent_errors_q3 = zeros(num_taus,1);

num_iters_to_converge_median = zeros(num_taus,1); 
num_iters_to_converge_q1 = zeros(num_taus,1); 
num_iters_to_converge_q3 = zeros(num_taus,1); 

total_runtimes_median = zeros(num_taus,1);
total_runtimes_q1 = zeros(num_taus,1);
total_runtimes_q3 = zeros(num_taus,1);

final_residuals_median = zeros(num_taus,1);
final_residuals_q1 = zeros(num_taus,1);
final_residuals_q3 = zeros(num_taus,1);

num_nnzs_median = zeros(num_taus,1); 
num_nnzs_q1 = zeros(num_taus,1); 
num_nnzs_q3 = zeros(num_taus,1); 

num_nnzs_x_median = zeros(num_taus,1); 
num_nnzs_x_q1 = zeros(num_taus,1); 
num_nnzs_x_q3 = zeros(num_taus,1); 


% calculate median and quartiles of percent errors for the different algoritms 
for ind=1:num_taus
    fprintf('calculating median, quartiles for end quantities at each tau: %d of %d\n', ind, num_taus);

    % we take median of each entry (1 to num_taus) over num_trials
    % this array holds num_trial numbers for each entry over which we take median
    tmp_array1 = zeros(num_trials,1);
    tmp_array2 = zeros(num_trials,1);
    tmp_array3 = zeros(num_trials,1);
    tmp_array4 = zeros(num_trials,1);
    tmp_array5 = zeros(num_trials,1);
    tmp_array6 = zeros(num_trials,1);

    for trial_num=1:num_trials
        % get arrays with num_taus variables for this particular trial
        myarr1 = cellarr_percent_errors_alg{trial_num};  
        myarr2 = cellarr_num_iters_to_converge_alg{trial_num};  
        myarr3 = cellarr_total_runtimes_alg{trial_num};  
        myarr4 = cellarr_final_residuals_alg{trial_num};  
        myarr5 = cellarr_num_nnzs_alg{trial_num};
        myarr6 = cellarr_num_nnzs_x{trial_num};

        % record entry for each trial
        tmp_array1(trial_num) = myarr1(ind);
        tmp_array2(trial_num) = myarr2(ind);
        tmp_array3(trial_num) = myarr3(ind);
        tmp_array4(trial_num) = myarr4(ind);
        tmp_array5(trial_num) = myarr5(ind);
        tmp_array6(trial_num) = myarr6(ind);
    end
    tmp_array1 = sort(tmp_array1);
    tmp_array2 = sort(tmp_array2);
    tmp_array3 = sort(tmp_array3);
    tmp_array4 = sort(tmp_array4);
    tmp_array5 = sort(tmp_array5);
    tmp_array6 = sort(tmp_array6);

    percent_errors_median(ind) = median(tmp_array1); 
    percent_errors_q1(ind) = median(tmp_array1(find(tmp_array1<median(tmp_array1)))); 
    percent_errors_q3(ind) = median(tmp_array1(find(tmp_array1>median(tmp_array1)))); 

    num_iters_to_converge_median(ind) = median(tmp_array2); 
    num_iters_to_converge_q1(ind) = median(tmp_array2(find(tmp_array2<median(tmp_array2)))); 
    num_iters_to_converge_q3(ind) = median(tmp_array2(find(tmp_array2>median(tmp_array2)))); 
    % if all entries of median are same this could be nan
    if isnan(num_iters_to_converge_q1(ind))
        num_iters_to_converge_q1(ind) = num_iters_to_converge_median(ind);
    end
    if isnan(num_iters_to_converge_q3(ind))
        num_iters_to_converge_q3(ind) = num_iters_to_converge_median(ind);
    end


    total_runtimes_median(ind) = median(tmp_array3); 
    total_runtimes_q1(ind) = median(tmp_array3(find(tmp_array3<median(tmp_array3)))); 
    total_runtimes_q3(ind) = median(tmp_array3(find(tmp_array3>median(tmp_array3)))); 
     % if all entries of median are same this could be nan
    if isnan(total_runtimes_q1(ind))
        total_runtimes_q1(ind) = total_runtimes_median(ind);
    end
    if isnan(total_runtimes_q3(ind))
        total_runtimes_q3(ind) = total_runtimes_median(ind);
    end


    final_residuals_median(ind) = median(tmp_array4); 
    final_residuals_q1(ind) = median(tmp_array4(find(tmp_array4<median(tmp_array4)))); 
    final_residuals_q3(ind) = median(tmp_array4(find(tmp_array4>median(tmp_array4)))); 
    % if all entries of median are same this could be nan
    if isnan(final_residuals_q1(ind))
        final_residuals_q1(ind) = final_residuals_median(ind);
    end
    if isnan(final_residuals_q3(ind))
        final_residuals_q3(ind) = final_residuals_median(ind);
    end


    num_nnzs_median(ind) = median(tmp_array5); 
    num_nnzs_q1(ind) = median(tmp_array5(find(tmp_array5<median(tmp_array5)))); 
    num_nnzs_q3(ind) = median(tmp_array5(find(tmp_array5>median(tmp_array5)))); 
    % if all entries of median are same this could be nan
    if isnan(num_nnzs_q1(ind))
        num_nnzs_q1(ind) = num_nnzs_median(ind);
    end
    if isnan(num_nnzs_q3(ind))
        num_nnzs_q3(ind) = num_nnzs_median(ind);
    end


    num_nnzs_x_median(ind) = median(tmp_array6); 
    num_nnzs_x_q1(ind) = median(tmp_array6(find(tmp_array6<median(tmp_array6)))); 
    num_nnzs_x_q3(ind) = median(tmp_array6(find(tmp_array6>median(tmp_array6)))); 
    % if all entries of median are same this could be nan
    if isnan(num_nnzs_x_q1(ind))
        num_nnzs_x_q1(ind) = num_nnzs_x_median(ind);
    end
    if isnan(num_nnzs_x_q3(ind))
        num_nnzs_x_q3(ind) = num_nnzs_x_median(ind);
    end
end

% save the data
save(vars_filename_str_driver);

