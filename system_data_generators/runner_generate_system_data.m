% generates a set of systems of the specified type
function runner_generate_system_data(dir_name, num_trials,  matrix_case, sparsityX, perturb_norms, zero_out_columns, noise_frac)

    for nt=1:num_trials
        fprintf('generating system %d of %d\n', nt, num_trials);
        
        [A,x,b,m,n,svdsA,noise,column_norms] = generate_system(matrix_case,sparsityX,perturb_norms,zero_out_columns,noise_frac);

        % save in matlab format
        fprintf('saving matlab system..\n');
        save([dir_name,'/system',num2str(nt),'.mat']);    

        % save A,x,b in matrix_market_format
        fprintf('saving matrix market system..\n');
        matrix_market_dir = [dir_name, '/matrix_market/system',num2str(nt),'/']
        system(['mkdir -p ', matrix_market_dir]);
        save_matrix_market_data(matrix_market_dir,A,x,b);
    end
end

