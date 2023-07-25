#=
# Markowitz Portfolio Optimization
This example shows how to solve a Markowitz portfolio optimization problem using
the QP interface of GeNIOS, acessed through `JuMP`.

Specifically, we want to solve the problem
$$
\begin{array}{ll}
\text{minimize}     & (1/2)\gamma x^T\Sigma x - \mu^Tx \\
\text{subject to}   & \mathbf{1}^Tx = 1 \\
                    & x \geq 0,
\end{array}
$$
where $\Sigma$ is the covariance matrix of the returns of the assets, $\mu$ is
the expected return of each asset, and $\gamma$ is a risk adversion parameter.
The variable $x$ represents the fraction of the total wealth invested in each
asset.
=#

using GeNIOS
using Random, LinearAlgebra, SparseArrays
using JuMP

#=
## Generating the problem data
Note that we generate the covariance matrix $\Sigma$ as a diagonal plus low-rank
matrix, which is a common model for financial data and referred to as a 'factor
model'
=#
Random.seed!(1)
k = 5
n = 100k
## Σ = F*F' + diag(d)
F = sprandn(n, k, 0.5)
d = rand(n) * sqrt(k)
μ = randn(n)
γ = 1;
Σ = F*F' + Diagonal(d)

#=
## Using JuMP
The easiest way to solve a QP using `GENIOS` is through `JuMP`.
We can define the problem as follows:
=#

model = Model(GeNIOS.Optimizer)
@variable(model, x[1:n])
@objective(model, Min, (γ/2)x'*Σ*x - μ'*x)
@constraint(model, sum(x) == 1)
@constraint(model, x .>= 0)
optimize!(model)

println("Optimal value: $(round(objective_value(model), digits=4))")

#=
Check out the portfolio optimization advanced example for performance improvements
=#