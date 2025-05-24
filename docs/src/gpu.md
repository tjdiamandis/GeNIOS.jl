# GPU Support

In this example, contributed by [Zack Li](https://gist.github.com/xzackli), we solve the Lasso problem with GeNIOS' `GenericInterface` and a custom `HessianOperator` which uses an NVIDIA GPU via [CUDA.jl](https://cuda.juliagpu.org/stable/).

```julia
using CUDA
using GeNIOS
using Random, LinearAlgebra, SparseArrays
using ProximalOperators

# the existing code in the generic interface tutorial
Random.seed!(1)
m, n = 5000, 10000
A = randn(m, n)
A .-= sum(A, dims=1) ./ m
normalize!.(eachcol(A))
xstar = sprandn(n, 0.1)
b = A*xstar + 1e-3*randn(m)
λ = 0.05*norm(A'*b, Inf)
solver_c = zeros(n)
tmp_arr = zeros(m)
struct HessianLasso{T, S <: AbstractMatrix{T}, V <: AbstractVector{T}} <: HessianOperator
    A::S
    vm::V
end
function LinearAlgebra.mul!(y, H::HessianLasso, x)
    mul!(H.vm, H.A, x)
    mul!(y, H.A', H.vm)
    return nothing
end

function update!(::HessianLasso, ::Solver)
    return nothing
end

# **********************************************************
# choose between Array or CuArray here, just comment one out
# **********************************************************
T = Float32
AT = CuArray{T} 
# AT = Array{T} 
A = AT(A)
b = AT(b)
λ = convert(eltype(A), λ)
solver_c = AT(solver_c)
tmp_arr = AT(tmp_arr)

params = (; A=A, b=b, tmp=tmp_arr, λ=λ)
function f(x, p)
    A, b, tmp = p.A, p.b, p.tmp
    mul!(tmp, A, x)
    @. tmp -= b
    return sum(w->w^2, tmp) / 2
end

function grad_f!(g, x, p)
    A, b, tmp = p.A, p.b, p.tmp
    mul!(tmp, A, x)
    @. tmp -= b
    mul!(g, A', tmp)
    return nothing
end

# **********************************************************
# NOTE: needed to convert the cache vector to AT
# **********************************************************
Hf = HessianLasso(A, AT(zeros(m))) 
g(z, p) = p.λ*sum(x->abs(x), z)

function prox_g!(v, z, ρ, p)
    λ = p.λ
    @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)
    v .= soft_threshold.(z, λ/ρ)
end


solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    g, prox_g!,             # g(z)
    I, solver_c;            # M, c: Mx + z = c
    params=params
)
res = solve!(solver; options=GeNIOS.SolverOptions{T,T}(
    relax=true, verbose=true, precondition=false, update_preconditioner=false))
rmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)
println("Final RMSE: $(round(rmse, digits=8))")
```





