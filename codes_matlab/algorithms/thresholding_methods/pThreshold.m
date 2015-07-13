function w=pThreshold(v, tau, p)
    n = length(v);
    w = v;
    for i=1:n
        w(i) = sign(v(i))*max(0,abs(v(i)) - tau*abs(v(i))^(p-1));
    end
end
