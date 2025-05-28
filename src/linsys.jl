abstract type HessianOperator end 

struct MLHessianOperator{
        T <: Real,
        V <: AbstractVector{T}, 
        M <: AbstractMatrix{T}, 
} <: HessianOperator
    w::V
    bdata::V
    Adata::M
    vN::V
    λ2::MVector{1,T}
end
function MLHessianOperator(Adata, bdata, λ2)
    N, _ = size(Adata)
    T = eltype(Adata)
    return MLHessianOperator(
        ones(N), bdata, Adata, zeros(T, N), MVector{1,T}(λ2)
    )
end

function LinearAlgebra.mul!(y, H::MLHessianOperator, x)
    # Assumes that w = f''(Adata*x - bdata) is updated
    # y = ∇²f(xᵏ)*x

    # y₁ = (Adataᵀ * Diagonal(w) * Adata) * x
    mul!(H.vN, H.Adata, x)
    @. H.vN *= H.w
    mul!(y, H.Adata', H.vN)
    
    # y = y + λ₂*I*x
    if H.λ2[1] > 0
        y .+= H.λ2[1] .* x
    end
    
    return nothing
end

# Quick fix: should not sketch the + λ₂I part of HessianOperator
# TODO: find a better work around
function RandomizedPreconditioners.NystromSketch(H::MLHessianOperator, r::Int; n=nothing, S=nothing)
    n = isnothing(n) ? size(H.Adata, 2) : n
    Y = S(undef, n, r)
    fill!(Y, 0.0)

    Ω = 1/sqrt(n) * randn(n, r)
    # TODO: maybe add a powering option here?
    for i in 1:r
        @views mul!(H.vN, H.Adata, Ω[:, i])
        @. H.vN *= H.w
        @views mul!(Y[:, i], H.Adata', H.vN)
    end
    
    ν = sqrt(n)*eps(norm(Y))
    @. Y = Y + ν*Ω

    Z = zeros(r, r)
    mul!(Z, Ω', Y)
    # Z[diagind(Z)] .+= ν                 # for numerical stability
    
    B = Y / cholesky(Symmetric(Z)).U
    U, Σ, _ = svd(B)
    Λ = Diagonal(max.(0, Σ.^2 .- ν))

    return NystromSketch(U, Λ)
end

function update!(H::MLHessianOperator, solver)
    @. H.w = solver.data.d2f(solver.pred)
    return nothing
end

struct ConicHessianOperator <: HessianOperator
    P
    σ
end
function LinearAlgebra.mul!(y, H::ConicHessianOperator, x)
    mul!(y, H.P, x)
    if H.σ > zero(typeof(H.σ))
        y .+= H.σ .* x
    end
    return nothing
end


function update!(::HessianOperator, solver)
    return nothing
end

# TODO: maybe the preconditioner should go in here?
struct LinearOperator{T,V <: AbstractVector{T},MT}
    M::MT
    ρ::MVector{1,T}
    Hf_xk::HessianOperator
    n::Int
    vm::V
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