#=
# Lasso
This example sets up a lasso regression problem with three different interfaces
provided by GeNIOS.

Specifically, we want to solve the problem
$$
\begin{array}{ll}
\text{minimize}     & (1/2)\|Ax - b\|_2^2 + \gamma \|x\|_1
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
γ = 0.05*norm(A'*b, Inf)

#=
## MLSolver interface
The easiest interface for this problem is the `MLSolver`, where we just need to
specify $f$ and the regularization parameters
=#
f(x) = 0.5*x^2 
df(x) = x
d2f(x) = 1.0
fconj(x) = 0.5*x^2
λ1 = γ
λ2 = 0.0
solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b; fconj=fconj)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, tol=1e-3, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")

#=
Note that we also defined the conjugate function of $f$, defined as

$$
f^*(y) = \sup_x \{yx - f(x)\},
$$
which allows us to use the dual gap as a stopping criterion (see appendix C of
our paper for a derivation). Specifying the conjugate function is optional, and
the solver will fall back to using the primal and dual residuals if it is not
specified.
=#


#=
## QPSolver interface
The `QPSolver` interface requires us to specify the problem in the form
$$
\begin{array}{ll}
\text{minimize}     & (1/2)x^TPx + q^Tx \\
\text{subject to}   &  l \leq Mx \leq u
\end{array}
$$
We introduce the new variable $t$ and inntroduce the constraint
$-t \leq x \leq t$ to enforce the $\ell_1$ norm. The problem then becomes

$$
\begin{array}{ll}
\text{minimize}     & (1/2)x^TA^TAx + b^TAx + \gamma \mathbf{1}^Tt \\
\text{subject to}   &  
\begin{bmatrix}0 \\ -\infty\end{bmatrix} 
\leq \begin{bmatrix} I & I \\ I & -I \end{bmatrix} \begin{bmatrix}x \\ t\end{bmatrix}  
\leq \begin{bmatrix}\infty \\ 0\end{bmatrix}
\end{array}
$$

=#

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

#=
## Generic interface
Finally, we use the generic interface, which provides a large amount of control
but is more complicated to use than the specialized interfaces demonstrated above.

First, we define the custom `HessianOperator`, which is used to solve the linear
system in the $x$-subproblem. Since the Hessian of the objective is simply $A^TA$,
this operator is simple for the Lasso problem. However, note that, since this
is a custom type, we can speed up the multiplication to be more efficient than
if we were to use $A^TA$ directly.

The `update!` function is called before the solution of the $x$-subproblem to 
update the Hessian, if necessary.
=#
struct HessianLasso{T, S <: AbstractMatrix{T}} <: HessianOperator
    A::S
    vm::Vector{T}
end
function LinearAlgebra.mul!(y, H::HessianLasso, x)
    mul!(H.vm, H.A, x)
    mul!(y, H.A', H.vm)
    return nothing
end

function update!(::HessianLasso, ::Solver)
    return nothing
end

#=
Now, we define $f$, its gradient, $g$, and its proximal operator.
=#
function f(x, A, b, tmp)
    mul!(tmp, A, x)
    @. tmp -= b
    return 0.5 * sum(w->w^2, tmp)
end
f(x) = f(x, A, b, zeros(m))
function grad_f!(g, x, A, b, tmp)
    mul!(tmp, A, x)
    @. tmp -= b
    mul!(g, A', tmp)
    return nothing
end
grad_f!(g, x) = grad_f!(g, x, A, b, zeros(m))
Hf = HessianLasso(A, zeros(m))
g(z, γ) = γ*sum(x->abs(x), z)
g(z) = g(z, γ)
function prox_g!(v, z, ρ)
    @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)
    v .= soft_threshold.(z, γ/ρ)
end

#=
Finally, we can solve the problem.
=#
solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    g, prox_g!,             # g(z)
    I, zeros(n);           # M, c: Mx + z = c
    ρ=1.0, α=1.0
)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")