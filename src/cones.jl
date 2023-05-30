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

function support(x::AbstractVector{T}, K::ProductCone) where {T}
    ind = 1
    ret = zero(T)
    for (i, Ki) in enumerate(K.cones)
        inds = ind:(ind + length(Ki) - 1) 
        ret += support(x[inds], Ki)
        ind += length(Ki)
    end
    return ret
end

function in_recession_cone(x::AbstractVector{T}, K::ProductCone, tol::T) where {T}
    ind = 1
    for (i, Ki) in enumerate(K.cones)
        inds = ind:(ind + length(Ki) - 1) 
       !in_recession_cone(x[inds], Ki, tol) && return false
        ind += length(Ki)
    end
    return true
end

# TODO: separate zero cone??
struct IntervalCone{T} <: Cone
    l::AbstractVector{T}
    u::AbstractVector{T}
end
function IntervalCone(l, u)
    length(l) != length(u) && throw(ArgumentError("l and u must have the same length"))
    any(l[i] > u[i], i in 1:length(l)) && throw(ArgumentError("l cannot be greater than u"))
    return IntervalCone(l, u)
end

Base.length(K::IntervalCone) = length(K.l)

function project!(x::AbstractVector{T}, K::IntervalCone{T}, y::AbstractVector{T}) where {T}
    @. x = clamp(y, K.l, K.u)
    return nothing
end

function support(x::AbstractVector{T}, K::IntervalCone{T}) where {T}
    @assert length(x) == length(K)
    
    # TODO: this is a hack
    if all(isinf, K.l) && all(isinf, K.u)
        return Inf
    end

    acc = 0.0
    @simd for i in 1:length(K)
        if x[i] > 0
            acc += K.u[i]*x[i]
        else
            acc += K.l[i]*x[i]
        end
    end
    return acc
end

function in_recession_cone(x::AbstractVector{T}, K::IntervalCone{T}, tol::T) where {T}
    for i in 1:length(K)
        if isinf(K.u[i]) && isinf(K.l[i])
            continue
        elseif isinf(K.u[i])
            x[i] < -tol && return false
        elseif isinf(K.l[i])
            x[i] > tol && return false
        else
            abs(x[i]) > tol && return false
        end
    end
    return true
end