%  performs soft thresholding on a vector
%  x[j] = (z[j]>tau) ? z[j]-tau : ((z[j]<-tau)? z[j]+tau : 0);
function w=softThreshold(v, tau)
    n = length(v);
    w = v;
    for i=1:n
        if v(i) > tau
            w(i) = v(i) - tau;
        elseif v(i) < -tau
            w(i) = v(i) + tau;
        else
            w(i) = 0;
        end
    end
end

