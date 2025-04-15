```@meta
CurrentModule = GeNIOS
```

# GeNIOS.jl
`GeNIOS` is an efficient large-scale operator splitting solver for convex 
optimization problems.

### Documentation Contents:
```@contents
Pages = ["index.md", "method.md", "guide.md", "api.md"]
Depth = 1
```
##### Examples:
```@contents
Pages = [
    "examples/constrained-ls.md",
    "examples/huber.md",
    "examples/lasso.md",
    "examples/logistic.md",
    "examples/portfolio.md",
]
Depth = 1
```

#### Advanced Usage:
```@contents
Pages = [
    "advanced/lasso.md",
    "advanced/portfolio.md",
    "advanced/signal.md"
]
Depth = 1
```


## Overview

GeNIOS solves convex optimization problems of the form

```math
\begin{array}{ll}
\text{minimize}     & f(x) + g(z) \\
\text{subject to}   & Mx + z = c,
\end{array}
```
where $x \in \mathbb{R}^n$ and $z \in \mathbb{R}^m$ are the optimization variables.
The function $f$ is assumed to be smooth, with known first and second derivatives,
and the function $g$ must have a known proximal operator.

Compared to conic form programs, the form we use in GeNIOS facilitates custom 
subroutines that often provide significant speedups. To ameliorate the extra 
complexity, we provide a few interfaces that take advantage of special problem
structure.

### Interfaces

#### QP interface
Many important problems in machine learning, finance, operations research, and control
can be formulated as QPs. Examples include
- Lasso regression
- Portfolio optimization
- Trajectory optimization
- Model predictive control
- And many more...

GeNIOS accepts QPs of the form
```math
\begin{array}{ll}
\text{minimize}     & (1/2)x^TPx + q^Tx \\
\text{subject to}   & l \leq Mx \leq u,
\end{array}
```
which can be constructed using
```julia
solver = GeNIOS.QPSolver(P, q, M, l, u)
```

#### ML interface
In machine learning problems, we can take advantage of additional structure.
In our `MLSolver`, we assume the problem is of the form
```math
\begin{array}{ll}
\text{minimize}     & \sum_{i=1}^m f(a_i^Tx - b_i) + (1/2)\lambda_2\|x\|_2^2 + \lambda_1\|x\|_1,
\end{array}
```
where $f$ is the per-sample loss function. Let $A$ be a matrix with rows $a_i$
and $b$ be a vector with entries $b_i$. The `MLSolver` is constructed via
```julia
solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b)
```
where `df`, and `d2f` are scalar functions giving the first and second derivative
of $f$ respectively.

In the future, we may extend this interface to allow constraints on $x$, but for
now, you can use the generic interface to specify constraints in machine learning
problems with non-quadratic objective functions.


#### Generic interface
For power users, we expose a fully generic interface for the problem
```math
\begin{array}{ll}
\text{minimize}     & f(x) + g(z) \\
\text{subject to}   & Mx + z = c.
\end{array}
```
Users must specify $f$, $\nabla f$, $\nabla^2 f$ (via a `HessianOperator`), 
$g$, the proximal operator of $g$, $M$, and $c$. We provide full details of these
functions in the [User Guide](@ref). Also checkout the [Lasso](@ref) example to 
see a problem written in all three ways.

### Algorithm
GeNIOS follows a similar approach to OSQP [^1], solving the convex optimization
problem using ADMM [^2]. Note that the problem form of GeNIOS is, however, more 
general than conic form solvers. The key algorithmic differences lies in the 
$x$-subproblem of ADMM. Instead of solving this problem exactly, GeNIOS solves a
quadratic approximation to this problem. Recent work [^3] shows that this 
approximation does not harm ADMM's convergence rate. The subproblem is then solved
via the iterative conjugate gradient method with a randomized preconditioner [^4].
GeNIOS also incorporates several other heuristics from the literature. Please
see [our paper]() for additional details.

## Quick Start
The JuMP interface is the easiest way to use GeNIOS.
A simple Markowitz portfolio example is below.
```julia
using JuMP, GeNIOS
using Random, SparseArrays, LinearAlgebra
Random.seed!(1)

k, n = 5, 100
# Σ = F*F' + diag(d)
F = sprandn(n, k, 0.5)
d = rand(n) * sqrt(k)
Σ = F*F' + diagm(d)
μ = randn(n)

# max μᵀx - (γ/2)*xᵀΣx, s.t. x ≥ 0 and 1ᵀx = 1
model = Model(GeNIOS.Optimizer)
@variable(model, x[1:n] >= 0)
@constraint(model, sum(x) == 1)
@objective(model, Max, sum(μ .* x) - (1/2)*x'*Σ*x)
optimize!(model)

xv = value.(x)
ret = dot(μ, xv) |> w->round(w, digits=3)
risk = sqrt(xv'*Σ*xv) |> w->round(w, digits=3)
println("Return:\t$ret\nRisk:\t$risk")

```

However, the native interfaces can be called directly by specifying the problem
data. Check out [Markowitz Portfolio Optimization, Three Ways](@ref) to see how
to solve this problem significantly faster.

The [User Guide](@ref) contains a full explanation of the solver parameter
options. Check out the examples as well.


## References
[^1]: Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). [OSQP: An operator splitting solver for quadratic programs](https://osqp.org). 

[^2]: Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). [Distributed optimization and statistical learning via the alternating direction method of multipliers](https://stanford.edu/~boyd/admm.html).

[^3]: Frangella, Z., Zhao, S., Diamandis, T., Stellato, B., & Udell, M. (2023). [On the (linear) convergence of Generalized Newton Inexact ADMM](https://arxiv.org/abs/2302.03863).

[^4]: Frangella, Z., Tropp, J. A., & Udell, M. (2021). [Randomized Nyström Preconditioning](https://arxiv.org/abs/2110.02820).