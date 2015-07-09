function [A,x,b,m,n,svdsA,noise,column_norms] = generate_system(matrix_case,sparsityX,perturb_norms,zero_out_columns,noise_frac)

    if matrix_case == 1
        % gaussian random matrix
        m = 300;
        n = 1500;
        A = sign(randn(m,n))*sqrt(1/m);
        x = full(sprand(n,1,sparsityX));
        for i=1:length(x)
            if x(i) ~= 0
                low = -2;
                high = 2;
                if mod(i,2) == 0
                    x(i) = -2 + 4*rand; % pick x(i) in [low,high] 
                else
                    x(i) = -10 + 20*rand; % alternate between high and low values
                end
            end
        end
        b = A*x;
        S = svds(A,min(m,n));

    elseif matrix_case == 2
        % singular values from 100 to 1 (well conditioned example)
        m = 250;
        n = 1000;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end
        if p>1
           S = linspace(100,1,p);
        else
           S = [1];
        end
        S = diag(S);
        A = U*S*V';

        % construct x
        x = full(sprand(n,1,sparsityX));
        for i=1:length(x)
            if x(i) ~= 0
                low = -2;
                high = 2;
                if mod(i,2) == 0
                    x(i) = -2 + 4*rand; % pick x(i) in [low,high] 
                elseif mod(i,3) == 0
                    x(i) = -20 + 40*rand; % pick x(i) in [low,high] 
                else
                    x(i) = -10 + 20*rand; % alternate between high and low values
                end
            end
        end
        % construct b vector
        b = A*x;

    elseif matrix_case == 3
        % singular values fall of more rapidly, more ill-conditioned example 
        m = 300;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end
        if p>1
           S = linspace(500,0.1,p);
        else
           S = [1];
        end
        S = diag(S);
        A = U*S*V';

        % construct x
        x = sprand(n,1,sparsityX);
        for i=1:length(x)
            if x(i) ~= 0
                low = -2;
                high = 2;
                x(i) = -2 + 4*rand; % pick x(i) in [low,high] 
            end
        end
        % construct b vector
        b = A*x;

    elseif matrix_case == 4
        % small logspaced singular values from 10 to 1e-5 - ill-conditioned
        m = 300;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end
        if p>1
           S = logspace(1,-3,p);
        else
           S = [1];
        end
        S = diag(S);
        A = U*S*V';

        % construct x
        x = sprand(n,1,sparsityX);
        for i=1:length(x)
            if x(i) ~= 0
                low = -2;
                high = 2;
                x(i) = -2 + 4*rand; % pick x(i) in [low,high] 
            end
        end


    % well conditioned staircase
    elseif matrix_case == 10
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        S = diag(linspace(100,0.1,p));

        A = U*S*V';
 
        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        mystep_vals = [-50:5:50];
        mystep_vals_ind = 1;
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,70) == 0
                bl = round(10*rand); % block length
                val = mystep_vals(mystep_vals_ind);
                mystep_vals_ind = mystep_vals_ind + 1;
                if mystep_vals_ind < 1 || mystep_vals_ind > length(mystep_vals)
                    mystep_vals_ind = 1;
                end
                if ((i + bl) < length(x)) && ((i-bl) > 1)
                    for bl_ind=-bl:bl
                        x(i+bl_ind) = val;
                    end
                end
            end
        end
        x = real(x); % filter out any imaginay stuff

    elseif matrix_case == 11
        % matrix conditioned like GN matrix
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        S = diag(linspace(100,0.1,p));

        A = U*S*V';
 
        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        mystep_vals = [-50:5:50];
        mystep_vals_ind = 1;
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,70) == 0
                bl = round(10*rand); % block length
                val = mystep_vals(mystep_vals_ind);
                mystep_vals_ind = round(length(mystep_vals)*rand);
                if mystep_vals_ind < 1 || mystep_vals_ind > length(mystep_vals)
                    mystep_vals_ind = 1;
                end
                if ((i + bl) < length(x)) && ((i-bl) > 1)
                    for bl_ind=-bl:bl
                        x(i+bl_ind) = val;
                    end
                end
            end
        end
        x = real(x); % filter out any imaginay stuff

    % ill conditioned staircase 
    elseif matrix_case == 12
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        ts = 1:p;
        svdsA = 1000*exp(-(5*ts-p)/p);
        S = diag(svdsA); 

        A = U*S*V';
 
        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        mystep_vals = [-50:5:50];
        mystep_vals_ind = 1;
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,70) == 0
                bl = round(10*rand); % block length
                val = mystep_vals(mystep_vals_ind);
                mystep_vals_ind = mystep_vals_ind + 1;
                if mystep_vals_ind < 1 || mystep_vals_ind > length(mystep_vals)
                    mystep_vals_ind = 1;
                end
                if ((i + bl) < length(x)) && ((i-bl) > 1)
                    for bl_ind=-bl:bl
                        x(i+bl_ind) = val;
                    end
                end
            end
        end
        x = real(x); % filter out any imaginay stuff

    elseif matrix_case == 13
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        ts = 1:p;
        svdsA = 1000*exp(-(5*ts-p)/p);
        S = diag(svdsA); 

        A = U*S*V';
 
        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        mystep_vals = [-50:5:50];
        mystep_vals_ind = 1;
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,70) == 0
                bl = round(10*rand); % block length
                val = mystep_vals(mystep_vals_ind);
                mystep_vals_ind = round(length(mystep_vals)*rand);
                if mystep_vals_ind < 1 || mystep_vals_ind > length(mystep_vals)
                    mystep_vals_ind = 1;
                end
                if ((i + bl) < length(x)) && ((i-bl) > 1)
                    for bl_ind=-bl:bl
                        x(i+bl_ind) = val;
                    end
                end
            end
        end
        x = real(x); % filter out any imaginay stuff

      % well conditioned unit grid
      elseif matrix_case == 14
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        svdsA = linspace(100,0.1,p);
        S = diag(svdsA);

        A = U*S*V';

        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,50) == 0
                x(i) = 1;
            end
        end
        x = real(x); % filter out any imaginay stuff


    % ill conditioned unit grid
    elseif matrix_case == 15
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        ts = 1:p;
        svdsA = 1000*exp(-(5*ts-p)/p);
        S = diag(svdsA);

        A = U*S*V';

        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,50) == 0
                x(i) = 1;
            end
        end
        x = real(x); % filter out any imaginary stuff


    % well conditioned unit two way grid
    elseif matrix_case == 16
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        svdsA = linspace(100,0.1,p);
        S = diag(svdsA);

        A = U*S*V';

        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,50) == 0
                x(i) = 1;
                x(i-1) = -1;
            end
        end
        x = real(x); % filter out any imaginay stuff

    % ill conditioned unit two way grid
    elseif matrix_case == 17
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        ts = 1:p;
        svdsA = 1000*exp(-(5*ts-p)/p);
        S = diag(svdsA);

        A = U*S*V';

        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,50) == 0
                x(i) = 1;
                x(i-1) = -1;
            end
        end
        x = real(x); % filter out any imaginay stuff


    % well conditioned one way grid with non unit components (of different amplitudes)
    elseif matrix_case == 18
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        svdsA = linspace(100,0.1,p);
        S = diag(svdsA);

        A = U*S*V';

        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,50) == 0
                rand_val1 = rand;
                rand_val2 = rand;
                if rand_val1 > 0.5
                    val = 1 + 0.3*rand_val2;
                else
                    val = 1 - 0.3*rand_val2;
                end
                x(i) = val;
            end
        end
        x = real(x); % filter out any imaginay stuff


    % ill conditioned one way grid with non unit components (of different amplitudes)
    elseif matrix_case == 19
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        ts = 1:p;
        svdsA = 1000*exp(-(5*ts-p)/p);
        S = diag(svdsA);

        A = U*S*V';

        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        for i=1:length(x)
            %if xref(i) ~= 0
            if mod(i,50) == 0
                rand_val1 = rand;
                rand_val2 = rand;
                if rand_val1 > 0.5
                    val = 1 + 0.3*rand_val2;
                else
                    val = 1 - 0.3*rand_val2;
                end
                x(i) = val;
            end
        end
        x = real(x); % filter out any imaginay stuff


    % well conditioned one way grid with non unit components (of different amplitudes) and non-uniform spacing
    elseif matrix_case == 20
        m = 1000;
        n = 1500;

        p = min(m,n);
        if m >= n
           [U, temp] = qr(randn(m,n),0);
           [V, temp] = qr(randn(n));
        else
           [U, temp] = qr(randn(m));
           [V, temp] = qr(randn(n,m),0);
        end

        svdsA = linspace(100,0.1,p);
        S = diag(svdsA);

        A = U*S*V';

        % construct x
        fprintf('building special x..\n');
        xref = sprand(n,1,sparsityX);
        x = zeros(n,1);
        for i=1:length(x)
            %if xref(i) ~= 0
            rand_val1 = rand;
            rand_val2 = 20*rand;
            if (rand_val1 < 0.5 && mod(i,50 + round(rand_val2)) == 0) || (rand_val1 >= 0.5 && mod(i,50 - round(rand_val2)) == 0)
                rand_val1 = rand;
                rand_val2 = rand;
                if rand_val1 > 0.5
                    val = 1 + 0.3*rand_val2;
                else
                    val = 1 - 0.3*rand_val2;
                end
                x(i) = val;
            end
        end
        x = real(x); % filter out any imaginay stuff

    % end all the else ifs
    end

    whos m n x

    m = size(A,1);
    n = size(A,2);

    % possibly perturb column norms
    if perturb_norms == 1
        for ind=1:n
            A(:,ind) = A(:,ind)*rand;
        end
    end

    % possibly zero out some column
    if zero_out_columns == 1
        for ind=1:n
            rand_val = rand;
            if rand_val < 0.40 % so we zero out 40 percent of the columns!
                v = A(:,ind);
                A(:,ind) = zeros(size(v,1),size(v,2));
            end
        end
    end
        
    % once A has been finalized, compute rhs 
    b = A*x;

    % find norm(A) - the largest singular value
    if perturb_norms == 2 || zero_out_columns == 2 % this takes way too long!! only if needed..
        fprintf('computing svds of A after modifications..\n');
        svdsA = svds(A,min(m,n)); % first few svds
    else
        svdsA = diag(S);
    end
    normA = max(max(svdsA)); % take the largest as the norm

    % scale A and b
    A = A/(1.1*normA);
    b = b/(1.1*normA);

    % add some Gaussian noise
    temp=randn(m,1);
    noise = noise_frac*norm(b)/norm(temp)*temp;
    b = b + noise;

    % compute column norms
    column_norms = zeros(n,1);
    fprintf('computing column norms\n');
    for i=1:n
        column_norms(i) = norm(A(:,i),2);
    end
    fprintf('done\n');
end

