function [x_grid, theta_grid, err_vals] = JoinedApproxErrorVals(pade_order, dx_wl, dz_wl, theta_max, n, approx_type)
max_spc_val = sin(pi*dz_wl*sin(theta_max * pi/180))^2 / ((2*pi*dz_wl)^2);
theta_grid = linspace(0, theta_max, n);
x_grid = -sin(pi*dz_wl*sin(theta_grid * pi/180)).^2 / ((2*pi*dz_wl)^2);
f = chebfun(@(x) exp(1i*2*pi*dx_wl*(sqrt(1+(1/(2*pi*dz_wl)^2 * acosh(1+(2*pi*dz_wl)^2 * x/2)^2))-1)),[-max_spc_val, 0]);
if strcmp(approx_type, 'chebpade')
    [p,q] = chebpade(f, pade_order(1), pade_order(2));
    r = p/q;
    err_f = abs(f - r);
    err_vals = err_f(x_grid);
else
    if strcmp(approx_type, 'ratinterp')
        [p,q] = ratinterp(f, pade_order(1), pade_order(2));
        r = p/q;
        err_f = abs(f - r);
        err_vals = err_f(x_grid);
    else
        if strcmp(approx_type, 'aaa')
            [r, pol, ~, zer, ~, ~, ~, ~] = aaa(f, 'degree', pade_order(1));
            rr = polyval(poly(zer), x_grid) ./ polyval(poly(pol), x_grid);
            rr = rr / rr(1) * r(0);
            err_vals = abs(f(x_grid) - rr);
        else
            syms x;
            fun_sym = exp(1i*2*pi*dx_wl*(sqrt(1+(1/(2*pi*dz_wl)^2 * acosh(1+(2*pi*dz_wl)^2 * x/2)^2))-1));
            padenm = pade(fun_sym, x, 0, 'Order', pade_order);
            err_vals = double(subs(abs(fun_sym - padenm), x, x_grid));
        end
    end
end
end