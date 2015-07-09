%  performs fivta thresholding on a vector as in the acha paper
function [w,L]=fivtaThreshold(v, rho, tau)
    L = 0;
    n = length(v);
    w = v;
    for i=1:n
        if v(i) > - rho && v(i) < rho
            w(i) = 0;
            L = L+1;
        else
            if v(i) >= tau
                w(i) = v(i) - (2*rho - tau);
            elseif v(i) >= rho && v(i) < tau
                w(i) = 2*(v(i) - rho);
            elseif v(i) > -tau && v(i) <= -rho
                w(i) = 2*(v(i) + rho);
            else
                w(i) = v(i) + (2*rho - tau);
            end
        end
    end
end

