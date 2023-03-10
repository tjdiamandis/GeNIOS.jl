abstract type HessianOperator end 

function update!(::HessianOperator, ::AbstractVector)
    return nothing
end

# TODO: maybe the preconditioner should go in here?
struct LinearOperator{T}
    A
    ρ::MVector{1,T}
    Hf_xk::HessianOperator
    n::Int
    vm::AbstractVector{T}
end
function LinearOperator(A, ρ::T, H::HessianOperator, m, n) where {T}
    return LinearOperator(A, MVector{1,T}(ρ), H, n, zeros(T, m))
end

function LinearAlgebra.mul!(y::AbstractVector{T}, M::LinearOperator{T}, x::AbstractVector{T}) where {T}
    # Assumes that ∇²f(xᵏ) is updated
    # y = ∇²f(xᵏ)*x
    mul!(y, M.Hf_xk, x)
    
    # y = Aᵀ(Ax)*ρ + y*1
    mul!(M.vm, M.A, x)
    mul!(y, M.A', M.vm, M.ρ[1], one(T))

    return nothing
end
Base.size(M::LinearOperator) = (M.n, M.n)
Base.eltype(::LinearOperator{T}) where {T} = T