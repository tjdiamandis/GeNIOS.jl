#=
# Lasso
This example uses the `LassoSolver` to solve a lasso regression problem.
We also show how to use the `MLSolver`.

The lasso regression problem is
$$
\begin{array}{ll}
\text{minimize}     & (1/2)\|Ax - b\|_2^2 + \lambda \|x\|_1.
\end{array}
$$
=#

using GeNIOS
using Random, LinearAlgebra, SparseArrays

#=
## Generating the problem data
=#
Random.seed!(1)
m, n = 200, 400
A = randn(m, n)
A .-= sum(A, dims=1) ./ m
normalize!.(eachcol(A))
xstar = sprandn(n, 0.1)
b = A*xstar + 1e-3*randn(m)
λ = 0.05*norm(A'*b, Inf)

#=
## LassoSolver interface
The easiest interface for this problem is the `LassoSolver`, where we just need to
specify the regularization parameter (in addition to the problem data).
=#
λ1 = λ
solver = GeNIOS.LassoSolver(λ1, A, b)
res = solve!(solver; options=GeNIOS.SolverOptions(use_dual_gap=true, dual_gap_tol=1e-4, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")

#=
## MLSolver interface
Under the hood, this is just a wrapper around the `MLSolver` interface.
This interface is more general, and allows us to specify the per-sample loss
used in the machine learning problem. Specifically, it solves problems with the
form
$$
\begin{array}{ll}
\text{minimize}     & \sum_{i=1}^N f(a_i^Tx - b_i) + \lambda_1 \|x\|_1 + (\lambda_2/2) \|x\|_2^2.
\end{array}
$$
It's easy to see that the lasso problem is a special case.
=#
f(x) = 0.5*x^2 
fconj(x) = 0.5*x^2
λ1 = λ
λ2 = 0.0
solver = GeNIOS.MLSolver(f, λ1, λ2, A, b; fconj=fconj)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, dual_gap_tol=1e-3, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")

#=
Note that we also defined the conjugate function of $f$, defined as

$$
f^*(y) = \sup_x \{yx - f(x)\},
$$
which allows us to use the dual gap as a stopping criterion (see
[our paper]() for a derivation). Specifying the conjugate function is optional, and
the solver will fall back to using the primal and dual residuals if it is not
specified.
=#
