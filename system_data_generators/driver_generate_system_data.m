num_trials = 3;


run_label = 'well_conditioned_staircase';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 10; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name, num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);


run_label = 'ill_conditioned_staircase';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 12; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name, num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);


return;

run_label = 'well_conditioned_staircase_high_noise';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 10; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.5;
runner_generate_system_data(dir_name, num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);


run_label = 'well_conditioned_staircase_perturb_norms';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 10; 
sparsityX = 0.05;
perturb_norms=1; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name, num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);





run_label = 'well_conditioned_random_staircase';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 11; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);


run_label = 'well_conditioned_random_staircase_high_noise';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 11; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.5;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);



run_label = 'ill_conditioned_random_staircase';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 13; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);



run_label = 'well_conditioned_unit_grid_with_non_unit_components';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 18; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);


run_label = 'well_conditioned_unit_two_way_grid';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 16; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);



run_label = 'ill_conditioned_unit_two_way_grid';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 17; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);



run_label = 'well_conditioned_unit_two_way_grid_high_noise';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 16; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.5;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);


run_label = 'well_conditioned_unit_grid_with_non_unit_components_high_noise';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 18; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.5;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);



run_label = 'ill_conditioned_unit_grid_with_non_unit_components';
dir_name = ['../data/system_data/', run_label];
system(['mkdir -p ', dir_name]);
system_num = 5;
matrix_case = 19; 
sparsityX = 0.05;
perturb_norms=0; zero_out_columns = 0;
noise_frac = 0.1;
runner_generate_system_data(dir_name,num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac);

