close all; clear all;

% plot quantities recorded by driver for different runs and algorithms

%%
%% WELL CONDITIONED STAIRCASE
%%

plotter_dir = '../data/codes_matlab/plotters/';
system(['mkdir -p ', plotter_dir]);

% set plot settings
residuals_y_lim = 150;
num_nnzs_y_lim = 1000;
percent_errors_y_lim = 100;
num_iters_y_lim = 300;
total_iters_y_lim = 3000;
total_runtimes_y_lim = 10;
settings_file = '../data/codes_matlab/plotters/settings1.mat';
save(settings_file);

% set driver file and image dir
driver_data_file = '../data/codes_matlab/drivers/well_conditioned_staircase/fista.mat';
image_dir = '../images/codes_matlab/well_conditioned_staircase/fista/';

% call plotter
runner_plotter(driver_data_file, image_dir, settings_file);


% set driver file and image dir
driver_data_file = '../data/codes_matlab/drivers/well_conditioned_staircase/fivta.mat';
image_dir = '../images/codes_matlab/well_conditioned_staircase/fivta/';

% call plotter
runner_plotter(driver_data_file, image_dir, settings_file);

