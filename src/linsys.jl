abstract type HessianOperator end 

struct MLHessianOperator{
        T <: Real,
        V <: AbstractVector{T}, 
        M <: AbstractMatrix{T}, 
        F <: Function
} <: HessianOperator
    w::V
    bdata::V
    Adata::M
    df2::F
    vN::V
    λ2::T
end
function MLHessianOperator(Adata, bdata, df2, λ2)
    N, n = size(Adata)
    return MLHessianOperator(zeros(N), bdata, Adata, df2, zeros(eltype(Adata), N), λ2)
end

function LinearAlgebra.mul!(y, H::MLHessianOperator, x)
    # Assumes that w = f''(Adata*x - bdata) is updated
    # y = ∇²f(xᵏ)*x

    # y₁ = (Adataᵀ * Diagonal(w) * Adata) * x
    mul!(H.vN, H.Adata, x)
    @. H.vN *= H.w
    mul!(y, H.Adata', H.vN)
    
    # y = y + λ₂*I*x
    H.λ2 > 0 && (y .+= H.λ2 .* x)
    return nothing
end

function update!(H::MLHessianOperator, solver)
    @. H.w = H.df2(solver.pred)
    return nothing
end

struct ConicHessianOperator <: HessianOperator
    P
end
LinearAlgebra.mul!(y, H::ConicHessianOperator, x) = mul!(y, H.P, x)


function update!(::HessianOperator, solver)
    return nothing
end

# TODO: maybe the preconditioner should go in here?
struct LinearOperator{T}
    M
    ρ::MVector{1,T}
    Hf_xk::HessianOperator
    n::Int
    vm::AbstractVector{T}
end
function LinearOperator(M, ρ::T, H::HessianOperator, m, n) where {T}
    return LinearOperator(M, MVector{1,T}(ρ), H, n, zeros(T, m))
end

function LinearAlgebra.mul!(y::AbstractVector{T}, M::LinearOperator{T}, x::AbstractVector{T}) where {T}
    # Assumes that ∇²f(xᵏ) is updated
    # y = ∇²f(xᵏ)*x
    mul!(y, M.Hf_xk, x)
    
    # y = Mᵀ(Mx)*ρ + y*1
    mul!(M.vm, M.M, x)
    mul!(y, M.M', M.vm, M.ρ[1], one(T))

    return nothing
end
Base.size(M::LinearOperator) = (M.n, M.n)
Base.eltype(::LinearOperator{T}) where {T} = T

function adaptive_nystrom_sketch(
    A,
    n::Int,
    r0::Int;
    condition_number=false,
    ρ=1e-4,
    r_inc_factor=2.0,
    tol=1e-6,
    q_norm=10,
    verbose=false
)
    cache = (
        u=zeros(n),
        v=zeros(n),
        Ahat_mul=zeros(n)
    )
    r = r0
    Ahat = nothing
    error_metric = Inf
    while error_metric > tol && r < n
        # k = round(Int, k_factor*r)
        Ahat = RP.NystromSketch(A, r; n=n)
        # Ahat = RP.NystromSketch(A, k, r, SketchType; check=check, q=q_sketch)
        if condition_number
            error_metric = (Ahat.Λ[end] + ρ) / ρ - 1
            verbose && @info "κ = $error_metric, r = $r"
        else
            error_metric = estimate_norm_E(A, Ahat; q=q_norm, cache=cache)
            verbose && @info "||E|| = $error_metric, r = $r"
        end
        r = round(Int, r_inc_factor*r)
    end
    return Ahat
end

# Power method to estimate ||A - Ahat|| (specialized for Symmetric)
function estimate_norm_E(A, Ahat::NystromSketch{T}; q=10, cache=nothing) where {T <: Number}
    n = size(Ahat, 2)
    if !isnothing(cache)
        u, v = cache.u, cache.v
    else
        u, v = zeros(T, n), zeros(T, n)
        cache = (Ahat_mul=zeros(T, n),)
    end
    
    randn!(v)
    normalize!(v)
    
    Ehat = Inf
    for _ in 1:q
        # v = (A - Ahat)*u
        mul!(u, Ahat, v; cache=cache.Ahat_mul)
        mul!(cache.Ahat_mul, A, v)
        u .-= cache.Ahat_mul
        # mul!(u, A, v, 1.0, -1.0)
        Ehat = dot(u, v)
        normalize!(u)
        v .= u

    end
    
    return Ehat
end