function [x_grid, theta_grid, err_vals] = ExpApproxErrorVals(pade_order, dx_wl, theta_max, n, approx_type)
%EXPAPPROXERROR Summary of this function goes here
%   Detailed explanation goes here
max_spc_val = sin(theta_max * pi/180)^2;
theta_grid = linspace(0, theta_max, n);
x_grid = -sin(theta_grid * pi/180).^2;
if strcmp(approx_type, 'chebpade')
    f = chebfun(@(x) exp(1i*2*pi*dx_wl*(sqrt(1+x)-1)),[-max_spc_val, 0]);
    [p,q] = chebpade(f, pade_order(1), pade_order(2));
    r = p/q;
    err_f = abs(f - r);
    err_vals = err_f(x_grid);
else
    if strcmp(approx_type, 'ratinterp')
        f = chebfun(@(x) exp(1i*2*pi*dx_wl*(sqrt(1+x)-1)),[-max_spc_val, 0]);
        [p,q] = ratinterp(f, pade_order(1), pade_order(2));
        r = p/q;
        err_f = abs(f - r);
        err_vals = err_f(x_grid);
    else
        if strcmp(approx_type, 'aaa')
            x = chebfun('x', [-max_spc_val 0]);
            f = exp(1i*2*pi*dx_wl*(sqrt(1+x)-1));
            [r, pol, ~, zer, ~, ~, ~, ~] = aaa(f, 'degree', pade_order(1));
            rr = polyval(poly(zer), x_grid) ./ polyval(poly(pol), x_grid);
            rr = rr / rr(1) * r(0);
            err_vals = abs(f(x_grid) - rr);
        else
            syms x;
            fun_sym = exp(1i*2*pi*dx_wl*(sqrt(1+x)-1));
            padenm = pade(fun_sym, x, 0, 'Order', pade_order);
            err_vals = double(subs(abs(fun_sym - padenm), x, x_grid));
        end
    end
end
end