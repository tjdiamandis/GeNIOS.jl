# Algorithm

Please see our paper [GeNIOS: an efficient large-scale operator splitting solver 
for convex composite optimization]() for full algorithmic details. This section
provides a very brief high-level sketch.

`GeNIOS` solves problems of the form
```math
\begin{array}{ll}
\text{minimize}     & f(x) + g(z) \\
\text{subject to}   & Mx + z = c,
\end{array}
```
where $f$ and $g$ are assumed to be convex.

## ADMM iteration
The main algorithm of `GeNIOS` is a variant of ADMM (see[^1] for a survey).
We introduce a (scaled) dual variable $u$ corresponding to the linear equality
constraint. The updates for the primal and dual variables are then
```math
\begin{aligned}
    x^{k+1} &= \argmin_{x} \left(f(x) + (\rho / 2)\|Mx - z^k - c + u^k\|_2^2 \right) \\
    z^{k+1} &= \argmin_{z} \left(g(z) + (\rho / 2)\|Mx^{k+1} - z - c + u^k\|_2^2 \right) \\
    u^{k+1} &= u^k + Mx^{k+1} - z^{k+1} - c. 
\end{aligned}
```
The core idea behind `GeNIOS` is to solve the $x$- and $z$-subproblems inexactly,
which significantly speeds up iteration time. For the $x$-subproblem, we use two 
approximation. First, we approximate $f$ around the current iterate with a
second-order Taylor expansion. Then, we solve the subproblem---which is now a
simple linear system solve---inexactly using the conjugate gradient method.
In addition, we used a randomized preconditioner (discussed in the next section),
in this subproblem to improve convergence. In the $z$-subproblem, we assume
access to an inexact proximal operator.

See [our paper]() for precise definitions of the inexactness that is required in
the subproblem solves.


## Fast linear system solves with Nyström PCG
After approximation, the $x$-subproblem only requires solving the linear system
```math
\left(\nabla^2 f(x^k) + \rho M^TM + \sigma I\right) x = (\nabla^2 f(x^k) + \sigma I) x^k - \nabla f(x^k) + \rho M^T (z^k + c - u^k).
```
The parameter $\sigma$ is chosen so that this system is always positive definite.
We use the randomized Nyström preconditioner[^2] to condition the spectrum of the
left hand size matrix in this problem. Empirically, we find that this preconditioner
often significantly improves convergence, as most problems with large amounts of
data have low-rank structure.


## References
[^1]: Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). [Distributed optimization and statistical learning via the alternating direction method of multipliers](https://stanford.edu/~boyd/admm.html).

[^2]: Frangella, Z., Tropp, J. A., & Udell, M. (2021). [Randomized Nyström Preconditioning](https://arxiv.org/abs/2110.02820).