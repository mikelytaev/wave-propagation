# Origin of the transport term $\varphi_\xi = i\sqrt B\,\varphi - \tfrac14 B^{-1}B_\xi\,\varphi + \dots$

The term comes from requiring the one-way equation to stay consistent with the
two-way equation (11) when $B$ changes with range. To first order the correction is
$-\tfrac14 B^{-1}B_\xi\varphi$, and it is exactly the term that conserves the energy
flux.

## 1. Factorizing a range-dependent equation

We start from eq. (11),

$$
\varphi_{\xi\xi} + B\varphi = 0,
$$

where $B(\xi)$ is an operator in $\eta$ that changes slowly with $\xi$. We look for a
one-way equation

$$
\varphi_\xi = i\Lambda(\xi)\,\varphi .
$$

Differentiating it once more, and substituting it back:

$$
\varphi_{\xi\xi} = i\Lambda_\xi\varphi + i\Lambda\varphi_\xi = (i\Lambda_\xi - \Lambda^2)\,\varphi .
$$

For this to match $\varphi_{\xi\xi} = -B\varphi$, $\Lambda$ has to satisfy the Riccati
equation

$$
\Lambda^2 - i\Lambda_\xi = B .
$$

- **When $B$ does not depend on $\xi$:** $\Lambda = \sqrt B$ solves it exactly. This is
  the usual PE.
- **When $B$ depends on $\xi$:** the term $i\Lambda_\xi$ is no longer zero. We expand in
  the slowness of the variation, $\Lambda = \Lambda_0 + \Lambda_1 + \dots$:
  - Order 0: $\Lambda_0^2 = B$, so $\Lambda_0 = \sqrt B$.
  - Order 1: $\Lambda_0\Lambda_1 + \Lambda_1\Lambda_0 = i(\sqrt B)_\xi$. Treating the
    operators as commuting at this order,
    $$\Lambda_1 = \frac i2(\sqrt B)^{-1}(\sqrt B)_\xi = \frac i4 B^{-1}B_\xi .$$

Hence

$$
\varphi_\xi = i\Lambda\varphi = i\sqrt B\,\varphi - \frac14 B^{-1}B_\xi\,\varphi + \dots
$$

The dots stand for second-order terms ($B_{\xi\xi}$ and $B_\xi^2$) and for the
commutators $[B, B_\xi]$, which arise because $B$ and $B_\xi$ are operators.

Section 4 works out this expansion in detail, including the operator case and the
second order.

## 2. What the term does: amplitude transport and flux

The new term is a logarithmic derivative,

$$
\frac14 B^{-1}B_\xi = \partial_\xi \log B^{1/4}.
$$

In the scalar case the solution is the WKB form

$$
\varphi = B^{-1/4}\exp\Big(i\int\sqrt B\,d\xi\Big).
$$

Differentiating this reproduces both terms. The exponential gives $i\sqrt B\varphi$,
and the prefactor $B^{-1/4}$ gives $-\tfrac14 B^{-1}B_\xi\varphi$.

Equivalently, set $\chi = B^{1/4}\varphi$. To the same order the correction then
cancels,

$$
\chi_\xi = i\sqrt B\,\chi ,
$$

so $\chi$ is propagated by the exact exponential and $\|\chi\|$ is conserved.
$\|\chi\|^2 = \|B^{1/4}\varphi\|^2$ is the energy flux through a line
$\xi=\text{const}$. To see this, write the flux as

$$
\operatorname{Im}\langle\varphi,\varphi_\xi\rangle = \operatorname{Re}\langle\varphi,\sqrt B\varphi\rangle = \|B^{1/4}\varphi\|^2
$$

for the propagating part.

This gives the flux-conserving step (15): map to $\chi$ with $B_n^{1/4}$, propagate
with the exponential, and map back with $B_{n+1}^{-1/4}$:

$$
\varphi_{n+1} = B_{n+1}^{-1/4}\, e^{i\Delta\xi\sqrt{B_{n+1/2}}}\, B_n^{1/4}\,\varphi_n .
$$

The plain step $\varphi_{n+1} = e^{i\Delta\xi\sqrt B}\varphi_n$ drops the term. It
conserves $\|\varphi\|^2$ instead, which is the wrong quantity.

## 3. Why it matters here but not in the usual PE

- **Physical picture.** For a local plane wave at angle $\theta$ to the marching
  direction, $\sqrt B \approx k h_\xi\cos\theta$, so $B^{1/4}\propto(\cos\theta)^{1/2}$.
  The flux through a line $\xi=\text{const}$ is the amplitude squared times
  $\cos\theta$. If $\theta$ changes from $\theta_1$ to $\theta_2$, flux conservation
  requires the amplitude to change by
  $$\left(\frac{\cos\theta_1}{\cos\theta_2}\right)^{1/2},$$
  the WKB factor quoted in the paper.
- **Flat ground.** $B$ changes with range only through $n^2$, which varies very slowly
  and almost uniformly in angle. $B_\xi/B\sim1/\ell$ with $\ell\gg\lambda$, so the term
  is dropped.
- **Terrain-following coordinates.** The coordinate lines rotate with height, so a beam
  changes its angle $\theta$ relative to the local $\xi$ direction as it crosses the
  transition layer. The change is order one, not slow, so dropping the term causes an
  amplitude error. In the slope tests it was 2–6% for beams reflected at 40–50°.
- **Where it stops mattering.** Above the layer $B$ is independent of $\xi$. There
  $B_\xi=0$, both pair brackets in (15) are the identity, and the scheme reduces to the
  flat-ground step. That is why the NLBC still applies.

## 4. The expansion of $\Lambda$ in detail

The expansion of $\Lambda$ is a WKB expansion of the outgoing Dirichlet-to-Neumann
operator. Order 0 gives the local wavenumber, order 1 the amplitude transport, order 2
a phase correction. Backscatter is not in it at any order.

### 4.1 Setup

The Riccati equation

$$
\Lambda^2 - i\Lambda_\xi = B
$$

is exact. To make "slowly varying" precise, let $B$ depend on a slow range
$X=\varepsilon\xi$, so that $\partial_\xi=\varepsilon\partial_X$. Here
$\varepsilon\sim1/(k\ell)$, with $\ell$ the length over which $B$ changes. Expand

$$
\Lambda = \Lambda_0 + \varepsilon\Lambda_1 + \varepsilon^2\Lambda_2 + \dots
$$

Substituting and collecting powers of $\varepsilon$ ($'$ denotes $\partial_X$):

| order | equation |
|---|---|
| $\varepsilon^0$ | $\Lambda_0^2 = B$ |
| $\varepsilon^1$ | $\Lambda_0\Lambda_1 + \Lambda_1\Lambda_0 = i\Lambda_0'$ |
| $\varepsilon^2$ | $\Lambda_0\Lambda_2 + \Lambda_2\Lambda_0 = i\Lambda_1' - \Lambda_1^2$ |
| $\varepsilon^n$ | $\Lambda_0\Lambda_n + \Lambda_n\Lambda_0 = i\Lambda_{n-1}' - \sum_{j=1}^{n-1}\Lambda_j\Lambda_{n-j}$ |

Each order is the same linear equation for the new term, with a right-hand side built
from the lower orders.

### 4.2 Order 0: choosing the branch

$\Lambda_0^2=B$ has two square roots, and the choice decides the direction of the wave:

- **Propagating spectrum** ($B>0$): $\operatorname{Re}\sqrt B>0$. With time dependence
  $e^{-i\omega t}$, $e^{+i\xi\sqrt B}$ travels toward $+\xi$.
- **Evanescent spectrum** ($B<0$): $\operatorname{Im}\sqrt B>0$, so the field decays
  in $\xi$.

$\Lambda_0=-\sqrt B$ would be the backward wave. Choosing $+\sqrt B$ is where the
one-way assumption enters.

### 4.3 Order 1: amplitude transport

The equation $\Lambda_0\Lambda_1+\Lambda_1\Lambda_0=i(\sqrt B)'$ is a Sylvester
equation. It has a unique solution because $\Lambda_0$ and $-\Lambda_0$ share no
eigenvalues: every $\sqrt\mu$ has a positive real or imaginary part.

**If $B$ and $B'$ commute** (the scalar case, or slow mode variation):

$$
\Lambda_1 = \frac i2(\sqrt B)^{-1}(\sqrt B)' = \frac i4 B^{-1}B' .
$$

This is the transport term $-\tfrac14B^{-1}B_\xi\varphi$.

**General operator case.** Work in the eigenbasis of $B(\xi)$, with
$Bv_j=\mu_jv_j$. The derivative of a matrix function in that basis is a divided
difference (the Daleckii–Krein formula):

$$
(\sqrt B)'_{jk} = \frac{B'_{jk}}{\sqrt{\mu_j}+\sqrt{\mu_k}} .
$$

The Sylvester equation is then diagonal in that basis:

$$
\boxed{(\Lambda_1)_{jk} = \frac{i\,B'_{jk}}{(\sqrt{\mu_j}+\sqrt{\mu_k})^2}}
$$

- **Diagonal** ($j=k$): the element is $iB'_{jj}/(4\mu_j)$, the familiar transport
  term for each mode.
- **Off-diagonal** ($j\ne k$): coupling between forward modes, that is, energy
  exchanged between waves at different angles. This is what the "commutator" dots in
  the transport equation stand for.

Both modes in the denominator are forward, so it never becomes small. This coupling is
regular, unlike forward–backward coupling.

### 4.4 Order 2: a phase correction

In the commuting case $\Lambda_2=(2\sqrt B)^{-1}(i\Lambda_1'-\Lambda_1^2)$, with

$$
i\Lambda_1' = -\frac{B''}{4B} + \frac{B'^2}{4B^2},\qquad
\Lambda_1^2 = -\frac{B'^2}{16B^2},
$$

which gives

$$
\boxed{\Lambda_2 = \frac{1}{2\sqrt B}\left(\frac{5B'^2}{16B^2} - \frac{B''}{4B}\right)}
$$

As a check, with $k=\sqrt B$ this is the standard second-order WKB term
$3k'^2/(8k^3)-k''/(4k^2)$. For real $B>0$ it is real, so it corrects the phase, not the
amplitude.

Relative sizes: $\Lambda_1/\Lambda_0\sim B'/(4B^{3/2})\sim1/(k\ell)$, and
$\Lambda_2/\Lambda_0\sim1/(k\ell)^2$.

### 4.5 What the expansion cannot capture

- **It is asymptotic, not convergent.** The exact outgoing operator, the one that also
  accounts for waves reflected from further ahead, differs from
  $\Lambda_0+\varepsilon\Lambda_1+\dots$ by terms like $e^{-c\,k\ell}$. These are smaller
  than every power of $\varepsilon$, which is why backscatter from a smooth medium is
  "beyond all orders". In the scalar case it is the coupling in the exact
  forward/backward system: with $\varphi=B^{-1/4}(a\,e^{iS}+b\,e^{-iS})$,
  $S=\int\sqrt B\,d\xi$,
  $$a_\xi=\frac{B_\xi}{4B}e^{-2iS}b,\qquad b_\xi=\frac{B_\xi}{4B}e^{2iS}a,$$
  so the first-order backscatter is
  $b(\xi)\approx-\int_\xi^\infty\frac{B_\xi}{4B}e^{2iS}a\,d\xi'$, an oscillatory
  integral. For a sharp change in $B$, such as a cliff, it is not small.
- **It breaks down at turning points.** All terms carry powers of $1/\mu_j$. When a
  component becomes perpendicular to the marching direction, $\mu_j\to0$ and the series
  diverges. This is the regime of total reflection and backward-going waves.

So the expansion describes forward propagation to all orders and contains no
backscatter, consistent with choosing the $+\sqrt B$ branch.

### 4.6 What the flux-conserving step keeps

The step $\varphi_{n+1}=B_{n+1}^{-1/4}e^{i\Delta\xi\sqrt{B_{n+1/2}}}B_n^{1/4}\varphi_n$
is equivalent to the generator

$$
i\sqrt B - B^{-1/4}(B^{1/4})' ,
$$

whose coupling term has, in the eigenbasis, the elements

$$
\frac{\mu_j^{-1/4}\,B'_{jk}}{(\mu_j^{1/4}+\mu_k^{1/4})(\sqrt{\mu_j}+\sqrt{\mu_k})} .
$$

- **Order 0:** taken at the midpoint, so it is second-order accurate in $\Delta\xi$.
- **Diagonal of order 1:** reproduced exactly, since for $j=k$ the element equals
  $B'_{jj}/(4\mu_j)$. This is the flux conservation.
- **Off-diagonal of order 1:** reproduced only approximately. Write
  $s_j=\mu_j^{1/4}$, $s_k=\mu_k^{1/4}$; the exact element is $B'_{jk}/(s_j^2+s_k^2)^2$.
  The difference is proportional to $s_k(s_k-s_j)$, so it vanishes for neighbouring modes and
  grows with the angle between the coupled waves.
- **Order 2 and above:** omitted. They are $O(1/(k\ell)^2)$ relative to the leading
  term.

The off-diagonal comparison is derived here and has not been measured in the code.
Since the exact-solution tests agree to about $10^{-3}$, it is evidently small in
practice. Implementing the exact $\Lambda_1$ coupling would be a possible refinement,
though not with the current tridiagonal rational factors.
