%  performs hard thresholding on a vector
function w=hardThreshold(v, tau)
    n = length(v);
    w = v;

    for i=1:n
        if abs(v(i)) < tau
            w(i) = 0;
        end
    end
end

