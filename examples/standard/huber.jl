#=
# Huber Fitting
This example sets up a $\ell_1$-regularized huber fitting problem
using the `MLSolver` interface provided by GeNIOS.
Huber fitting is a form of 'robust regression' that is less sensitive to 
outliers.

The huber loss function is defined as
$$
f^\mathrm{hub}(x) = \begin{cases}
\frac{1}{2}x^2 & \lvert x\rvert \leq 1 \\
|x| - \frac{1}{2} & \lvert x \rvert > 1
\end{cases}
$$

We want to solve the problem
$$
\begin{array}{ll}
\text{minimize}     & \sum_{i=1}^N f^\mathrm{hub}(a_i^T x - b_i) + \gamma \|x\|_1
\end{array}
$$
=#
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GeNIOS
using Random, LinearAlgebra, SparseArrays

#=
## Generating the problem data
=#
Random.seed!(1)
N, n = 200, 400
A = randn(N, n)
A .-= sum(A, dims=1) ./ N
normalize!.(eachcol(A))
xstar = sprandn(n, 0.1)
b = A*xstar + 1e-3*randn(N)

## Add outliers
b += 10*collect(sprand(N, 0.05))
γ = 0.05*norm(A'*b, Inf)

#=
## MLSolver interface
We just need to specify $f$ and the regularization parameters.
=#

## Huber problem: min ∑ fʰᵘᵇ(aᵢᵀx - bᵢ) + γ||x||₁
f(x) = abs(x) <= 1 ? 0.5*x^2 : abs(x) - 0.5
df(x) = abs(x) <= 1 ? x : sign(x)
d2f(x) = abs(x) <= 1 ? 1 : 0
λ1 = γ
λ2 = 0.0
solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=false, verbose=true))
rmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")

#=
### Automatic differentiation
We could have let the solver figure out the derivatives for us as well:
=#
f(x) = abs(x) <= 1 ? 0.5*x^2 : abs(x) - 0.5
λ1 = γ
λ2 = 0.0
solver = GeNIOS.MLSolver(f, λ1, λ2, A, b)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=false, verbose=true))
rmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")