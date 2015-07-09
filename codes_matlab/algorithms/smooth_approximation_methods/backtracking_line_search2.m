% alg 3.1 p. 37 in nocedal/wright book
% good choice of c and rho varies by matrix/signal
function alpha=backtracking_line_search2(A,At,AtA,b,tau,sigma,xn,p)
    alpha=approximate_line_search(A,At,AtA,b,tau,sigma,xn,p);
    c = 0.01;
    rho = 0.5;

    F1 = evalF(A,b,tau,sigma,xn + alpha*p);
    F2 = evalF(A,b,tau,sigma,xn);
    gradxn = gradF(A,At,b,tau,sigma,xn);

    while F1 > F2 + c*alpha*dot(gradxn,p)
        alpha = rho*alpha;
        F1 = evalF(A,b,tau,sigma,xn + alpha*p);
    end
end
