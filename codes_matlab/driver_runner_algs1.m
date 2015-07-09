% this top level script runs variopus algorithms on different 
% data sets 
addpath('algorithms/thresholding_methods/');
addpath('algorithms/smooth_approximation_methods/');
addpath('algorithms/other_methods/');
% how many trials are we doing for each data set type
% make sure this number of systems exists in ../data/
% this is the number we specified in system_data_generators 
num_trials = 3; 
num_taus = 10; % define at how many taus to sample along ||Ax-b|| vs tau curve

%%
%% WELL CONDITIONED STAIRCASE
%%

system_data_dir = '../data/system_data/well_conditioned_staircase/';
runner_data_dir = '../data/codes_matlab/runners/well_conditioned_staircase/';
driver_data_dir = '../data/codes_matlab/drivers/well_conditioned_staircase/';
system(['mkdir -p ', runner_data_dir]);
system(['mkdir -p ', driver_data_dir]);

alg_name = 'fista';
maxiters = 100;
TOL = 1e-8;
vars_filename_str_runner = [runner_data_dir,'fista.mat']; % filename for runner to save data to
vars_filename_str_driver = [driver_data_dir,'fista.mat']; % filename for runner to save data to
driver_Axbvstau_1alg(alg_name, system_data_dir, vars_filename_str_runner, vars_filename_str_driver, num_taus, num_trials, maxiters, TOL);


alg_name = 'fivta';
maxiters = 100;
TOL = 1e-8;
vars_filename_str_runner = [runner_data_dir,'fivta.mat']; % filename for runner to save data to
vars_filename_str_driver = [driver_data_dir,'fivta.mat']; % filename for runner to save data to
driver_Axbvstau_1alg(alg_name, system_data_dir, vars_filename_str_runner, vars_filename_str_driver, num_taus, num_trials, maxiters, TOL);


%%
%% ILL CONDITIONED STAIRCASE
%%

system_data_dir = '../data/system_data/ill_conditioned_staircase/';
runner_data_dir = '../data/codes_matlab/runners/ill_conditioned_staircase/';
driver_data_dir = '../data/codes_matlab/drivers/ill_conditioned_staircase/';
system(['mkdir -p ', runner_data_dir]);
system(['mkdir -p ', driver_data_dir]);

alg_name = 'fista';
maxiters = 100;
TOL = 1e-8;
vars_filename_str_runner = [runner_data_dir,'fista.mat']; % filename for runner to save data to
vars_filename_str_driver = [driver_data_dir,'fista.mat']; % filename for runner to save data to
driver_Axbvstau_1alg(alg_name, system_data_dir, vars_filename_str_runner, vars_filename_str_driver, num_taus, num_trials, maxiters, TOL);


alg_name = 'fivta';
maxiters = 100;
TOL = 1e-8;
vars_filename_str_runner = [runner_data_dir,'fivta.mat']; % filename for runner to save data to
vars_filename_str_driver = [driver_data_dir,'fivta.mat']; % filename for runner to save data to
driver_Axbvstau_1alg(alg_name, system_data_dir, vars_filename_str_runner, vars_filename_str_driver, num_taus, num_trials, maxiters, TOL);

