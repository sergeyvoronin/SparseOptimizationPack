% based on taylor series approximation 
function alpha=approximate_line_search(A,At,AtA,b,tau,sigma,xn,p)
    gradxn = gradF(A,At,b,tau,sigma,xn);
    Hessxn = hessianF(A,At,AtA,b,tau,sigma,xn);

    alpha = -dot(gradxn,p)/(dot(p,Hessxn*p));
end
