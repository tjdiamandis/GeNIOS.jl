#=
# Logistic Regression
This example sets up a $\ell_1$-regularized logistic regression problem
using the `MLSolver` interface provided by GeNIOS.

Specifically, we want to solve the problem
$$
\begin{array}{ll}
\text{minimize}     & \sum_{i=1}^N \log(1 + \exp(a_i^T x)) + \gamma \|x\|_1
\end{array}
$$
=#

using GeNIOS
using Random, LinearAlgebra, SparseArrays

#=
## Generating the problem data
=#
Random.seed!(1)
N, n = 2_000, 4_000
Ã = sprandn(N, n, 0.2)
@views [normalize!(Ã[:, i]) for i in 1:n-1]
Ã[:,n] .= 1.0

xstar = zeros(n)
inds = randperm(n)[1:100]
xstar[inds] .= randn(length(inds))
b̃ = sign.(Ã*xstar + 1e-1 * randn(N))
b = zeros(N)
A = Diagonal(b̃) * Ã

γmax = norm(0.5*A'*ones(N), Inf)
γ = 0.05*γmax

#=
## MLSolver interface
We just need to specify $f$ and the regularization parameters. We also define
the conjugate function $f^*$, defined as

$$
f^*(y) = \sup_x \{yx - f(x)\},
$$
to use the dual gap convergence criterion. 
=#
## Logistic problem: min ∑ log(1 + exp(aᵢᵀx)) + γ||x||₁
f2(x) = GeNIOS.log1pexp(x)
df2(x) = GeNIOS.logistic(x)
d2f2(x) = GeNIOS.logistic(x) / GeNIOS.log1pexp(x)
f2conj(x::T) where {T} = (one(T) - x) * log(one(T) - x) + x * log(x)
λ1 = γ
λ2 = 0.0
solver = GeNIOS.MLSolver(f2, df2, d2f2, λ1, λ2, A, b; fconj=f2conj)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, tol=1e-4, verbose=true))

#=
## Results
=#
rmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")
println("Dual gap: $(round(res.dual_gap, digits=8))")