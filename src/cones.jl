abstract type Cone end

struct ProductCone <: Cone
    cones::Vector{Cone}
end

function project!(x::AbstractVector{T}, K::ProductCone, y::AbstractVector{T}) where {T}
    ind = 1
    for (i, Ki) in enumerate(K.cones)
        inds = ind:(ind + length(Ki) - 1) 
        project!(x[inds], Ki, y[inds])
        ind += length(Ki)
    end
    return nothing
end

# TODO: separate zero cone??
struct IntervalCone{T} <: Cone
    l::AbstractVector{T}
    u::AbstractVector{T}
end
function IntervalCone(l, u)
    length(l) != length(u) && throw(ArgumentError("l and u must have the same length"))
    return IntervalCone(l, u)
end

Base.length(K::IntervalCone) = length(K.l)

function project!(x::AbstractVector{T}, K::IntervalCone{T}, y::AbstractVector{T}) where {T}
    @. x = clamp(y, K.l, K.u)
    return nothing
end