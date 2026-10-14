# Scale factors $h_\xi$, $h_\eta$ from the level-curve family $F(x,\eta)$

Given $F(x,\eta)$, two quantities must be found: the trajectory abscissa $X(\xi,\eta)$
and its spread $P=X_\xi$. The scale factors $h_\xi$, $h_\eta$ are then
the lengths of the two tangent vectors.

## 1. The map

A point $(\xi,\eta)$ goes to

$$
x = X(\xi,\eta),\qquad z = F\big(X(\xi,\eta),\eta\big).
$$

The second component holds because the point lies on the level curve $\eta$: its
height is $F$ evaluated at its own abscissa. $F(x,\eta)$ is known explicitly from
eq. (8) of the paper. $X(\xi,\eta)$ is the unknown.

## 2. The two tangent vectors

- **Along a level curve** ($\eta$ fixed, $\xi$ varying):
  $$\mathbf a_\xi = (X_\xi,\; Z_\xi) = (X_\xi,\; F_x X_\xi) = P\,(1,\,F_x),\qquad P\equiv X_\xi .$$
- **Along a trajectory** ($\xi$ fixed, $\eta$ varying), by the chain rule:
  $$\mathbf a_\eta = (X_\eta,\; Z_\eta) = (X_\eta,\; F_x X_\eta + F_\eta).$$

Here $F_x$ and $F_\eta$ are partial derivatives of the explicit function $F(x,\eta)$,
evaluated at $x=X(\xi,\eta)$.

## 3. Orthogonality gives an ODE for $X$

Setting the dot product of the trajectory tangent with $(1,F_x)$ to zero:

$$
X_\eta + F_x\,(F_x X_\eta + F_\eta) = 0
\quad\Longrightarrow\quad
X_\eta = -\frac{F_x F_\eta}{1+F_x^2} \equiv G(X,\eta).
$$

This is eq. (9). For each $\xi$ it is an ODE for the curve that crosses every level
curve at a right angle.

**Initial condition.** Above the layer the level curves are horizontal, so $G=0$ and
the trajectories are vertical. They are labelled by their abscissa there:
$X(\xi,\eta_L)=\xi$. Integrating downward from $\eta_L$ to $0$ gives the trajectories
inside the layer.

## 4. $h_\eta$: the length of the trajectory tangent

Substituting $X_\eta=G$:

$$
F_x X_\eta + F_\eta = F_\eta - \frac{F_x^2 F_\eta}{1+F_x^2} = \frac{F_\eta}{1+F_x^2},
$$

$$
X_\eta^2 + (F_x X_\eta + F_\eta)^2 = \frac{F_\eta^2\,(F_x^2+1)}{(1+F_x^2)^2} = \frac{F_\eta^2}{1+F_x^2}.
$$

Hence

$$
\boxed{h_\eta = \frac{F_\eta}{\sqrt{1+F_x^2}}}
$$

**Geometric meaning.** Neighbouring level curves are a vertical distance $F_\eta\,d\eta$
apart. The true distance between them is measured along the normal, which makes the
angle $\theta=\arctan F_x$ with the vertical, so the distance is
$F_\eta\,d\eta\cos\theta$. $h_\eta$ depends only on $F$ at the point, with no
integration needed.

## 5. $h_\xi$: the length of the level-curve tangent

$$
\boxed{h_\xi = P\sqrt{1+F_x^2}}
$$

Here $\sqrt{1+F_x^2}$ converts horizontal distance into arc length along the level
curve. $P=X_\xi$ is the horizontal distance between neighbouring
trajectories, relative to their spacing above the layer. $P$ measures how the
trajectories converge ($P<1$) or diverge ($P>1$) on their way down, and it is the only
non-local part of the metrics.

## 6. An ODE for $P$

Differentiate (9) with respect to $\xi$. $G$ depends on $\xi$ only through $X$, so

$$
\partial_\eta X_\xi = G_x(X,\eta)\,X_\xi
\quad\Longrightarrow\quad
P_\eta = G_x(X,\eta)\,P,\qquad P(\xi,\eta_L)=1 .
$$

This is eq. (10). Its solution is

$$
P = \exp\!\Big(-\int_\eta^{\eta_L} G_x\big(X(\xi,\eta'),\eta'\big)\,d\eta'\Big),
$$

so $P$ accumulates the convergence or divergence of the trajectory field along the
path. $G_x$ is the partial $x$-derivative of $G$ at fixed $\eta$:

$$
G_x = -\frac{F_{xx}F_\eta + F_x F_{x\eta}}{1+F_x^2} + \frac{2F_x^2 F_{xx} F_\eta}{(1+F_x^2)^2}.
$$

## 7. A consistency check

The Jacobian of the map is

$$
\det\begin{pmatrix} P & X_\eta \\ F_x P & F_x X_\eta + F_\eta \end{pmatrix} = P\,F_\eta ,
$$

and the product of the two scale factors gives the same thing:

$$
h_\xi h_\eta = P\sqrt{1+F_x^2}\cdot\frac{F_\eta}{\sqrt{1+F_x^2}} = P\,F_\eta ,
$$

as expected for orthogonal coordinates. The anisotropy is

$$
r = \frac{h_\eta}{h_\xi} = \frac{F_\eta}{P\,(1+F_x^2)} .
$$

## 8. How the code computes this

See `OrthogonalTerrainMap` in `pywaveprop/terrain_transform.py`: `family`, `_rhs`,
`_integrate` and `metrics`.

1. **Spectral derivatives.** At each $\eta$ level, one inverse FFT of $\hat u(\kappa,\eta)$
   from (8) gives $u$, $u_x=\mathcal F^{-1}[i\kappa\hat u]$,
   $u_{xx}=\mathcal F^{-1}[-\kappa^2\hat u]$, $u_\eta$ and $u_{x\eta}$ on a fine uniform
   $x$ grid, with
   $$\partial_\eta\hat u = \hat b\,\big(w'/L - |\kappa|\,w\big)\,e^{-|\kappa|\eta}.$$
   Then $F_x=u_x$ and $F_\eta=1+u_\eta$.
2. **Interpolation to the trajectories.** The trajectory abscissas $X$ are not grid
   points, so these fields are interpolated to the current $X$ (`np.interp` on the
   fine grid). $G$ and $G_x$ are then formed there.
3. **Integration.** $X$ and $P$ are advanced together with RK4, downward from
   $\eta_L$, for all $\xi$ columns at once, on coarse $\eta$ levels.
4. **Metrics on the computational grid.** On each coarse level,
   $h_\eta=F_\eta/\sqrt{1+F_x^2}$ and $h_\xi=P\sqrt{1+F_x^2}$ are evaluated. Cubic
   splines in $\eta$ then interpolate them to the computational nodes.

**Summary.** $h_\eta$ is local and closed-form. $h_\xi$ needs the trajectory ODE,
because orthogonality fixes how the trajectories spread.
