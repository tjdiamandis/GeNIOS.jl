The source files for all examples can be found in [/examples](https://github.com/tjdiamandis/GeNIOS.jl/tree/main/examples).
```@meta
EditURL = "../../../examples/advanced/portfolio.jl"
```

# Markowitz Portfolio Optimization, Three Ways
This example shows how to solve a Markowitz portfolio optimization problem using
both the `GenericSolver` and the `QPSolver`.

Specifically, we want to solve the problem
```math
\begin{array}{ll}
\text{minimize}     & (1/2)\gamma x^T\Sigma x - \mu^Tx \\
\text{subject to}   & \mathbf{1}^Tx = 1 \\
                    & x \geq 0,
\end{array}
```
where $\Sigma$ is the covariance matrix of the returns of the assets, $\mu$ is
the expected return of each asset, and $\gamma$ is a risk adversion parameter.
The variable $x$ represents the fraction of the total wealth invested in each
asset.

````@example portfolio
using GeNIOS
using Random, LinearAlgebra, SparseArrays
````

## Generating the problem data
Note that we generate the covariance matrix $\Sigma$ as a diagonal plus low-rank
matrix, which is a common model for financial data and referred to as a 'factor
model'

````@example portfolio
Random.seed!(1)
k = 5
n = 100k
# Σ = F*F' + diag(d)
F = sprandn(n, k, 0.5)
d = rand(n) * sqrt(k)
μ = randn(n)
γ = 1;
nothing #hide
````

## QPSolver interface
To use the `QPSolver`, we need to
specify $P$, $q$, $M$, $l$, and $u$. The Markowitz portfolio optimization problem
is equivalent to the following 'standard form' QP:
```math
\begin{array}{ll}
\text{minimize}     & (1/2)x^T(\gamma \Sigma) x + (-\mu)Tx \\
\text{subject to}   &
\begin{bmatrix}1 \\ 0\end{bmatrix}
\leq \begin{bmatrix} I \\ \mathbf{1}^T \end{bmatrix} x
\leq \begin{bmatrix}\infty \\ 1\end{bmatrix}
\end{array}
```

````@example portfolio
P = γ*(F*F' + Diagonal(d))
q = -μ
M = vcat(I, ones(1, n))
l = vcat(zeros(n), ones(1))
u = vcat(Inf*ones(n), ones(1))
solver = GeNIOS.QPSolver(P, q, M, l, u)
res = solve!(solver; options=GeNIOS.SolverOptions(eps_abs=1e-6))
println("Optimal value: $(round(solver.obj_val, digits=4))")
````

### Performance improvements
We can also define custom operators for $P$ and $M$ to speed up the computation.
For this, we will use `LinearMaps.jl`, which allows us to build these operators
without ever explicitly constructing the final matrices.

If we don't implement `size` for all objects passed to the solver, we need set
`check_dims` to `false`.

````@example portfolio
using LinearMaps
# P = γ*(F*F' + Diagonal(d))
F_lm = LinearMap(F)
P = γ*(F_lm*F_lm' + Diagonal(d))

# M = vcat(I, ones(1, n))
M = vcat(LinearMap(I, n), ones(1, n))

solver = GeNIOS.QPSolver(P, q, M, l, u; check_dims=false);
res = solve!(solver; options=GeNIOS.SolverOptions(eps_abs=1e-6));
println("Optimal value: $(round(solver.obj_val, digits=4))")
````

### Custom `P` and `M`
Of course, we can also define each needed operations ourselves, which may
give an additional speedup for some matrices. Here, we avoid allocations
for the intermediate product when multiplying by $F^T$ and then $F$.

````@example portfolio
struct FastP
    F
    d
    γ
    vk
end
function LinearAlgebra.mul!(y::AbstractVector, P::FastP, x::AbstractVector)
    mul!(P.vk, P.F', x)
    mul!(y, P.F, P.vk)
    @. y += P.d*x
    @. y *= P.γ
    return nothing
end
Pfast = FastP(F, d, γ, zeros(k))

struct FastM
    n::Int
end
Base.size(M::FastM) = (M.n+1, M.n)
Base.size(M::FastM, d::Int) = d <= 2 ? size(M)[d] : 1
function LinearAlgebra.mul!(y::AbstractVector, M::FastM, x::AbstractVector)
    y[1:M.n] .= x
    y[end] = sum(x)
    return nothing
end
LinearAlgebra.adjoint(M::FastM) = Adjoint{Float64, FastM}(M)
function LinearAlgebra.mul!(x::AbstractVector{T}, M::Adjoint{T, FastM}, y::AbstractVector{T}) where T <: Number
    @. x = y[1:M.parent.n] + y[end]
    return nothing
end
function LinearAlgebra.mul!(x::AbstractVector{T}, M::Adjoint{T, FastM}, y::AbstractVector{T}, α::T, β::T) where T <: Number
    @. x = α * ( y[1:M.parent.n] + y[end] ) + β * x
    return nothing
end

P = FastP(F, d, γ, zeros(k))
M = FastM(n)
solver = GeNIOS.QPSolver(P, q, M, l, u; check_dims=false);
res = solve!(solver; options=GeNIOS.SolverOptions(eps_abs=1e-6));
println("Optimal value: $(round(solver.obj_val, digits=4))")
````

## GenericSolver interface
The `GenericSolver` interface is more flexible and allows for some speedups via
an alternative problem splitting. We will solve the equivalent problem
```math
\begin{array}{ll}
\text{minimize}     & (1/2)\gamma x^T\Sigma x - \mu^Tx + I(z) \\
\text{subject to}   & -x + z = 0
\end{array}
```
where $I(z)$ is the indicator function for the set $\{z \mid z \ge 0 \text{ and } \mathbf{1}^Tz = 1\}$.
The gradient and Hessian of $f(x) = (1/2)\gamma x^T\Sigma x$ are easy to compute.
The proximal operator of $g(z) = I(z)$ is simply the projection on this set,
which can be solved via a one-dimensional root-finding problem (see
the appendix of [our paper]()).

````@example portfolio
params = (; F=F, d=d, μ=μ, γ=γ, tmp=zeros(k))

# f(x) = γ/2 xᵀ(FFᵀ + D)x - μᵀx
function f(x, p)
    F, d, μ, γ, tmp = p.F, p.d, p.μ, p.γ, p.tmp
    mul!(tmp, F', x)
    qf = sum(abs2, tmp)
    qf += sum(i->d[i]*x[i]^2, 1:length(x))

    return γ/2 * qf - dot(μ, x)
end

#  ∇f(x) = γ(FFᵀ + D)x - μ
function grad_f!(g, x, p)
    F, d, μ, γ, tmp = p.F, p.d, p.μ, p.γ, p.tmp
    mul!(tmp, F', x)
    mul!(g, F, tmp)
    @. g += d*x
    @. g *= γ
    @. g -= μ
    return nothing
end

# ∇²f(x) = γ(FFᵀ + D)
struct HessianMarkowitz{T, S <: AbstractMatrix{T}} <: HessianOperator
    F::S
    d::Vector{T}
    vk::Vector{T}
end
function LinearAlgebra.mul!(y, H::HessianMarkowitz, x)
    mul!(H.vk, H.F', x)
    mul!(y, H.F, H.vk)
    @. y += d*x
    return nothing
end
Hf = HessianMarkowitz(F, d, zeros(k))

# g(z) = I(z)
function g(z, p)
    T = eltype(z)
    return all(z .>= zero(T)) && abs(sum(z) - one(T)) < 1e-6 ? zero(T) : Inf
end
function prox_g!(v, z, ρ, p)
    z_max = maximum(w->abs(w), z)
    l = -z_max - 1
    u = z_max

    # bisection search to find zero of F
    while u - l > 1e-8
        m = (l + u) / 2
        if sum(w->max(w - m, zero(eltype(z))), z) - 1 > 0
            l = m
        else
            u = m
        end
    end
    ν = (l + u) / 2
    @. v = max(z - ν, zero(eltype(z)))
    return nothing
end

solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    g, prox_g!,             # g(z)
    I, zeros(n);            # M, c: Mx - z = c
    params=params
)
res = solve!(solver)
println("Optimal value: $(round(solver.obj_val, digits=4))")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

