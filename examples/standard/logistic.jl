#=
# Logistic Regression
This example sets up a $\ell_1$-regularized logistic regression problem
using the `LogisticSolver` and `MLSolver` interfaces.

Logistic regression is the problem
$$
\begin{array}{ll}
\text{minimize}     & \sum_{i=1}^N \log(1 + \exp(a_i^T x)) + \lambda \|x\|_1
\end{array}
$$
=#

using GeNIOS
using Random, LinearAlgebra, SparseArrays

#=
## Generating the problem data
=#
Random.seed!(1)
N, n = 200, 400
Ã = sprandn(N, n, 0.2)
Ã .-= sum(Ã, dims=1) ./ N
normalize!.(eachcol(Ã))
Ã[:,n] .= 1.0
xstar = zeros(n)
inds = randperm(n)[1:100]
xstar[inds] .= randn(length(inds))
b̃ = sign.(Ã*xstar + 1e-1 * randn(N))

A = Diagonal(b̃) * Ã
b = zeros(N)

λmax = norm(0.5*A'*ones(N), Inf)
λ = 0.05*λmax

#=
## Logistic Solver
The easiest way to solve this problem is to use our `LogisticSolver` interface.
=#
λ1 = λ
λ2 = 0.0
solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
res = solve!(solver; options=GeNIOS.SolverOptions(use_dual_gap=true, dual_gap_tol=1e-4, verbose=true))

#=
## MLSolver interface
Under the hood, the LogisticSolver is just a wrapper around the MLSolver interface.
We can see what it's doing below. 
To use the `MLSolver` interface,
we just need to specify $f$ and the regularization parameters. We also define
the conjugate function $f^*$, defined as

$$
f^*(y) = \sup_x \{yx - f(x)\},
$$
to use the dual gap convergence criterion. 
=#
## Logistic problem: min ∑ log(1 + exp(aᵢᵀx)) + λ||x||₁
f2(x) = GeNIOS.log1pexp(x)
f2conj(x::T) where {T} = (one(T) - x) * log(one(T) - x) + x * log(x)
λ1 = λ
λ2 = 0.0
solver = GeNIOS.MLSolver(f2, λ1, λ2, A, b; fconj=f2conj)
res = solve!(solver; options=GeNIOS.SolverOptions(use_dual_gap=true, dual_gap_tol=1e-4, verbose=true))

#=
## Results
=#
rmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")
println("Dual gap: $(round(res.dual_gap, digits=8))")