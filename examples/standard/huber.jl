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
λ = 0.05*norm(A'*b, Inf)

#=
## MLSolver interface
We just need to specify $f$ and the regularization parameters.
=#

## Huber problem: min ∑ fʰᵘᵇ(aᵢᵀx - bᵢ) + λ||x||₁
f(x) = abs(x) <= 1 ? 0.5*x^2 : abs(x) - 0.5
λ1 = λ
λ2 = 0.0
solver = GeNIOS.MLSolver(f, λ1, λ2, A, b)
res = solve!(solver; options=GeNIOS.SolverOptions(use_dual_gap=false))
rmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")

#=
## Duality gap
However, we can supply the conjugate function if we want to use the duality gap.
=#
fconj(y) = y > 1 ? Inf : y^2/2.0
solver = GeNIOS.MLSolver(f, λ1, λ2, A, b; fconj=fconj)
res = solve!(solver; options=GeNIOS.SolverOptions(use_dual_gap=true))
rmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")