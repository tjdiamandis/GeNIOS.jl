abstract type Cone end

# TODO: separate zero cone??
struct IntervalCone{T} <: Cone
    l::AbstractVector{T}
    u::AbstractVector{T}
end

function project!(x::AbstractVector{T}, K::IntervalCone{T}, y::AbstractVector{T}) where {T}
    @. x = clamp(y, K.l, K.u)
    return nothing
end