function [a, b, a0] = JoinedChebPadeCoefs(pade_order_num, pade_order_den, dx_wl, dz_wl, max_spc_val, type)
%EXPCHEBPADE Summary of this function goes here
%   Detailed explanation goes here
f = chebfun(@(x) exp(1i*2*pi*dx_wl*(sqrt(1+(1/(2*pi*dz_wl)^2 * acosh(1+(2*pi*dz_wl)^2 * x/2)^2))-1)),[-max_spc_val, 0]);
p = [];
q = [];
if type == "chebpade"
    [p, q] = chebpade(f, double(pade_order_num), double(pade_order_den));
    r = p/q;
    a0 = 1;
    a = -1 ./ roots(p, 'all').';
    b = -1 ./ roots(q, 'all').';
    abs_error = max(abs(f - r));
end
if type == "ratinterp"
    [p, q] = ratinterp(f, double(pade_order_num), double(pade_order_den));
    r = p/q;
    a0 = 1;
    a = -1 ./ roots(p, 'all').';
    b = -1 ./ roots(q, 'all').';
    abs_error = max(abs(f - r));
end
if type == "aaa"
    [r, pol, ~, zer, ~, ~, ~, ~] = aaa(f, 'degree', double(pade_order_den));
    a0 = r(0) / (polyval(poly(zer), 0) ./ polyval(poly(pol), 0));
    a = -1 ./ zer;
    b = -1 ./ pol; 
end
end