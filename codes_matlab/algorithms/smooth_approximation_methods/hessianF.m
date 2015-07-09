% computes the hessian matrix of F using the differential approximation 
% \hat\phi to |x|
function M=hessianF(A,At,AtA,b,tau,sigma,x)

    m = size(A,1); n = size(A,2);
    alpha = 2*sqrt(2/pi)*sigma;
    
    diag_vec = zeros(n,1);
    for j=1:n
        xj = x(j);
        %diag_vec(j) = xj*exp(-xj^2/(2*sigma^2)); 
        diag_vec(j) = exp(-xj^2/(2*sigma^2)) - 1/(2*sigma^2)*xj^2*exp(-xj^2/(2*sigma^2));
        if abs(xj) < 1e-8
            diag_vec(j) = 0;
        end
    end
    D = diag(diag_vec);
    
    M = 2*AtA + 4*tau*sqrt(2)/(sigma*sqrt(pi)) * D;
end

