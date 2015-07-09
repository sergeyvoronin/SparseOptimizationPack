% computes the gradient of F using the differential approximation \hat\phi to |x|
function g=gradF(A,At,b,tau,sigma,x)

    m = size(A,1); n = size(A,2);
    erf_vec = zeros(n,1);
    alpha = sqrt(2/pi)*(1/sigma);

    for j=1:n
        xj = x(j);
        erf_vec(j) = erf(xj/(sqrt(2)*sigma)) + alpha*xj*exp(-xj^2/(2*sigma^2)); 
    end
    g = 2*At*(A*x - b) + 2*tau*(erf_vec);

end
