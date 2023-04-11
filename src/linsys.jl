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