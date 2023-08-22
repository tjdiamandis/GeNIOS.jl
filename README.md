# GeNIOS.jl ("genie-o̅s")

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tjdiamandis.github.io/GeNIOS.jl/dev/)
[![Build Status](https://github.com/tjdiamandis/GeNIOS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tjdiamandis/GeNIOS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/tjdiamandis/GeNIOS.jl/branch/main/graph/badge.svg?token=1DKZD7FPW5)](https://codecov.io/gh/tjdiamandis/GeNIOS.jl)


## Overview

The GEneralized Newton Inexact Operator Splitting (`GeNIOS`) package contains an
experimental inexact ADMM solver for convex problems. We both approximate the
ADMM subproblems and then solve these subproblems inexactly. The algorithm is
inspired by [Frangella et al. 2023](https://arxiv.org/abs/2302.03863).

We provide interfaces for machine learning problems (`MLSolver`) and quadratic
programs (`QPSolver`). In addition, convex optimization problems may be specified
directly.

For more information, check out the [documentation](https://tjdiamandis.github.io/GeNIOS.jl/dev/).

## Quick  start, 3 ways
First, add the package locally.
```julia 
using Pkg; Pkg.add(url="https://github.com/tjdiamandis/GeNIOS.jl")
```

Generate some data
```julia
using GeNIOS, Random, LinearAlgebra

Random.seed!(1)
m, n = 200, 400
A = randn(m, n)
A .-= sum(A, dims=1) ./ m
normalize!.(eachcol(A))
xstar = sprandn(n, 0.1)
b = A*xstar + 1e-3*randn(m)
γ = 0.05*norm(A'*b, Inf)
```

Create a `LassoSolver` and solve:
```julia
λ1 = γ
solver = GeNIOS.LassoSolver(λ1, A, b)
res = solve!(solver; options=GeNIOS.SolverOptions(use_dual_gap=true, dual_gap_tol=1e-4, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")
```

### `MLSolver` interface
Under the hood, the `LassoSolver` is calling the `MLSolver` interface. We can
do this directly if we want to modify the per-sample loss function or regularization.
The equivalent problem, with the `MLSolver` interface, can be defined as follows.
```julia
f(x) = 0.5*x^2
df(x) = x
d2f(x) = 1.0
fconj(x) = 0.5*x^2
λ1 = γ
λ2 = 0.0
solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b; fconj=fconj)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, dual_gap_tol=1e-3, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")
```

Note that `fconj` is only required to use the dual gap as the convergence criterion.
In addition, instead of supplying `df` and `df2`, we can let the solver figure these 
out via automatic differentiation.
```julia
solver = GeNIOS.MLSolver(f, λ1, λ2, A, b)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=false, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")
```

### `QPSolver` interface
Since the lasso problem is a quadratic program, we can use the `QPSolver` interface
as well. Note that we must introduce an additional variable $t$ and the constraint
$-t \leq x \leq t$ to enforce the $\ell_1$ norm. The problem then becomes

```math
\begin{array}{ll}
\text{minimize}     & (1/2)x^TA^TAx + b^TAx + \gamma \mathbf{1}^Tt \\
\text{subject to}   &  
\begin{bmatrix}0 \\ -\infty\end{bmatrix} 
\leq \begin{bmatrix} I & I \\ I & -I \end{bmatrix} \begin{bmatrix}x \\ t\end{bmatrix}  
\leq \begin{bmatrix}\infty \\ 0\end{bmatrix}
\end{array}
```

The following code uses the `QPSolver`:
```julia
P = blockdiag(sparse(A'*A), spzeros(n, n))
q = vcat(-A'*b, γ*ones(n))
M = [
    sparse(I, n, n)     sparse(I, n, n);
    sparse(I, n, n)     -sparse(I, n, n)
]
l = [zeros(n); -Inf*ones(n)]
u = [Inf*ones(n); zeros(n)]
solver = GeNIOS.QPSolver(P, q, M, l, u)
res = solve!(
    solver; options=GeNIOS.SolverOptions(
        relax=true,
        max_iters=1000,
        eps_abs=1e-4,
        eps_rel=1e-4,
        verbose=true)
);
rmse = sqrt(1/m*norm(A*solver.xk[1:n] - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")
```

### `GenericSolver` interface
The generic solver included with `GeNIOS` is somewhat more involved to use.
Please check out the [corresponding section of the documentation](https://tjdiamandis.github.io/GeNIOS.jl/dev/examples/lasso/#Generic-interface)
for the example above.

## References
- Frangella, Z., Zhao, S., Diamandis, T., Stellato, B., & Udell, M. (2023). On the (linear) convergence of Generalized Newton Inexact ADMM. arXiv preprint arXiv:2302.03863.
