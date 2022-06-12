function [x_grid, err_vals] = ExpApproxErrorValsM(pade_order, dx_wl, min_spc_val, max_spc_val, n, approx_type)
%EXPAPPROXERROR Summary of this function goes here
%   Detailed explanation goes here
x_grid = linspace(min_spc_val, max_spc_val, n);
if strcmp(approx_type, 'ratinterp')
    f = chebfun(@(x) exp(1i*2*pi*dx_wl*(sqrt(1+x)-1)),[-max_spc_val, max_spc_val]);
    [p,q] = ratinterp(f, pade_order(1), pade_order(2));
    r = p/q;
    err_f = abs(f - r);
    err_vals = err_f(x_grid);
else
    syms x;
    fun_sym = exp(1i*2*pi*dx_wl*(sqrt(1+x)-1));
    padenm = pade(fun_sym, x, 0, 'Order', pade_order);
    err_vals = double(subs(abs(fun_sym - padenm), x, x_grid));
end
end
