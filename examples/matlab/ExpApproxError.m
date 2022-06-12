function [abs_error] = ExpApproxError(pade_order, dx_wl, theta_max, approx_type)
%EXPAPPROXERROR Summary of this function goes here
%   Detailed explanation goes here
max_spc_val = sin(theta_max * pi/180)^2;
if strcmp(approx_type, 'chebpade')
    f = chebfun(@(x) exp(1i*2*pi*dx_wl*(sqrt(1+x)-1)),[-max_spc_val, 0]);
    [p,q] = chebpade(f, pade_order(1), pade_order(2));
    r = p/q;
    abs_error = max(abs(f - r));
else
    syms x;
    fun_sym = exp(1i*2*pi*dx_wl*(sqrt(1+x)-1));
    padenm = pade(fun_sym, x, 0, 'Order', pade_order);
    abs_error = max(double(subs(abs(fun_sym - padenm), x, linspace(-max_spc_val, 0, 10))));
end
end
