% evaluate the approximated $\ell_1$ functional
function val=evalF(A,b,tau,sigma,x)
    %val = norm(A*x - b,2)^2 + 2*tau*norm(x,1);
    m = size(A,1); n = size(A,2);

    erf_vec = zeros(n,1);
    for j=1:n
        erf_vec(j) = x(j)*erf(x(j)/(sqrt(2)*sigma)) + sqrt(2/pi)*sigma*exp(-x(j)*x(j)/(2*sigma^2)) - sqrt(2/pi)*sigma; 
    end

    val = norm(A*x - b,2)^2 + 2*tau*sum(erf_vec);
end
