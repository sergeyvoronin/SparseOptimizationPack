function save_matrix_market_data(matrix_market_dir,A,x,b)

m = size(A,1); n = size(A,2);

% write to files
fprintf('write matrix A\n');
[i,j,k] = find(A);
num_nnz = length(k);
fp = fopen([matrix_market_dir,'/A.mtx'],'w');
for ind=1:num_nnz
    if ind==1
        fprintf(fp,'%%');
        fprintf(fp,'%%');
        fprintf(fp,'MatrixMarket matrix coordinate real general\n');
    elseif ind==2
        fprintf(fp,'\t%d\t%d\t%d\n', m,n,num_nnz-2);
    else
        fprintf(fp,'%d %d %f\n', i(ind),j(ind),k(ind));
    end
end
fclose(fp);


fprintf('write vector x\n');
fp = fopen([matrix_market_dir,'/x.txt'],'w');
fprintf(fp,'%d\n',n);
for ind=1:n
    fprintf(fp,'%f\n',x(ind));
end
fclose(fp);

fprintf('write vector b\n');
fp = fopen([matrix_market_dir,'/b.txt'],'w');
fprintf(fp,'%d\n',m);
for ind=1:m
    fprintf(fp,'%f\n',b(ind));
end
fclose(fp);

end
