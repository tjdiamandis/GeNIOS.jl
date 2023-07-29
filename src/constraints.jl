# Ax[inds] - c ∈ K
struct Constraint{
    T <: Real, 
    VEC <: AbstractVector{T},
    LM <: Union{AbstractVecOrMat{T}, UniformScaling}
}
    M::LM
    c::VEC
    K::Cone
    inds::Union{AbstractRange{Int}, Vector{Int}}
    function Constraint(
        M::LM,
        c::VEC,
        K::Cone,
        inds::Union{AbstractRange{Int}, Vector{Int}}
    ) where {
        T <: Real, 
        VEC <: AbstractVector{T},
        LM <: Union{AbstractVecOrMat{T}, UniformScaling},
    }
        if M isa UniformScaling 
            m = n = length(c)
        else 
            m, n = size(M)
        end
        (length(c) != m || length(K) != m || length(inds) != n) && 
            throw(DimensionMismatch("Dimension mismatch"))
        
        return new{T, VEC, LM}(M, c, K, inds)
    end
end
Base.length(c::Constraint) = length(c.K)
Base.size(c::Constraint) = (length(c.K), length(c.inds))

function build_constraints(constraints::Vector{T}, n::Union{Int,Nothing}=nothing) where {T <: Constraint}
    @inline function count_nnz(constraint::Constraint)
        M = constraint.M
        M = M isa UniformScaling ? sparse(M, size(constraint)...) : M
        return M isa AbstractSparseMatrix ? nnz(M) : length(M)
    end
    m = sum(length.(constraints))
    n = isnothing(n) ? maximum([maximum(c.inds) for c in constraints]) : n
    nnz_tot = sum(count_nnz, constraints)

    K = ProductCone([c.K for c in constraints])
    c = mapreduce(x -> getfield(x, :c), vcat, constraints)

    Is = Vector{Int}(undef, nnz_tot)
    Js = Vector{Int}(undef, nnz_tot)
    Vs = Vector{eltype(c)}(undef, nnz_tot)

    idx = 1
    idx_row = 1
    for constraint in constraints
        M = constraint.M
        M = M isa UniformScaling ? sparse(M, size(constraint)...) : M

        if M isa AbstractSparseMatrix
            nn = nnz(M)
            I, J, V = findnz(M)
            @views Is[idx:(idx + nn - 1)] .= I .+ idx_row .- 1
            @views Js[idx:(idx + nn - 1)] .= constraint.inds[J]
            @views Vs[idx:(idx + nn - 1)] .= V
        elseif M isa AbstractMatrix
            Mm, Mn = size(M)
            nn = Mm * Mn
            is = repeat(1:Mm, Mn)
            js = repeat(1:Mn, inner=Mm)
            @views Is[idx:(idx + nn - 1)] .= is .+ idx_row .- 1
            @views Js[idx:(idx + nn - 1)] .= constraint.inds[js]
            @views Vs[idx:(idx + nn - 1)] .= vec(M)
        else
            error("constraint type $(typeof(M)) not supported")
        end
        idx += nn
        idx_row += size(M, 1)
    end
    @assert idx == nnz_tot + 1
    M = sparse(Is, Js, Vs, m, n)

    return M, c, K
end


# TODO: add in matrix-free support XXX
# function build_block_linear_map(constraints::Vector{T}) where {T <: Constraint}
#     N = length(constraints)
#     m = sum(length.(constraints))
#     @inline function convert_to_lmap(c::Constraint)
#         M = c.M
#         m, n = size(c)
#         if M isa LinearMap
#             return  M
#         elseif M isa UniformScaling
#             return LinearMap(M, n)
#         elseif M isa AbstractVecOrMat
#             return LinearMap(M)
#         else
#             error("constraint type $(typeof(M)) not supported")
#         end
#     end

#     maps = convert_to_lmap.(c)
#     row_ranges = [c.inds for c in constraints]
#     col_ranges = Vector{UnitRange}(undef, N)
#     rows = ntuple(_ -> 1, N)
    
#     ind_curr = 1
#     for (idx, c) in enumerate(constraints)
#         col_ranges[idx] = ind_curr:(ind_curr + length(c) - 1)
#         ind_curr += length(c)
#     end
#     @show ind_curr
#     @assert ind_curr == m + 1

#     M = LinearMaps.BlockMap(
#         maps,
#         rows,
#         row_ranges,
#         col_ranges
#     )
#     return M
# end