#=
# Lasso, Three Ways
This example shows how to use the `MLSolver`, `QPSolver`, and the `GenericSolver`
interfaces to solve a lasso regression problem.

The lasso regression problem is
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
λ = 0.05*norm(A'*b, Inf)

#=
## MLSolver interface
The `MLSolver` interface requires the per-sample loss and (optionally) its
conjugate, if we want to use the dual gap as a stopping criterion (discussed in
our paper).
The conjugate function of $f$, defined as
$$
f^*(y) = \sup_x \{yx - f(x)\},
$$
Specifying the conjugate function is optional, and
the solver will fall back to using the primal and dual residuals if it is not
specified.
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
Note that the solver uses forward-mode automatic differentiation
to compute the first and second derivatives of $f$. We can supply these arguments
for additional speedup, if desired.
=#
df(x) = x
d2f(x) = 1.0
solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b; fconj=fconj)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, dual_gap_tol=1e-3, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")


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
q = vcat(-A'*b, λ*ones(n))
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
params = (; A=A, b=b, tmp=zeros(m), λ=λ)
function f(x, p)
    A, b, tmp = p.A, p.b, p.tmp
    mul!(tmp, A, x)
    @. tmp -= b
    return 0.5 * sum(w->w^2, tmp)
end

function grad_f!(g, x, p)
    A, b, tmp = p.A, p.b, p.tmp
    mul!(tmp, A, x)
    @. tmp -= b
    mul!(g, A', tmp)
    return nothing
end

Hf = HessianLasso(A, zeros(m))
g(z, p) = p.λ*sum(x->abs(x), z)

function prox_g!(v, z, ρ, p)
    λ = p.λ
    @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)
    v .= soft_threshold.(z, λ/ρ)
end

#=
Finally, we can solve the problem.
=#
solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    g, prox_g!,             # g(z)
    I, zeros(n);            # M, c: Mx + z = c
    params=params
)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")


#=
## ProximalOperators.jl
We could alternatively use `ProximalOperators.jl` to define the proximal operator
for g:
=#
using ProximalOperators
prox_func = NormL1(λ)
gp(x) = prox_func(x)
prox_gp!(v, z, ρ, p) = prox!(v, prox_func, z, ρ)

## We see that this give the same result
solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    gp, prox_gp!,           # g(z)
    I, zeros(n);            # M, c: Mx + z = c
    params=params
)
res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, verbose=true))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")