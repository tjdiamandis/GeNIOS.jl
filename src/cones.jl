abstract type Cone end
import Base.length

struct ProductCone <: Cone
    cones::Vector{Cone}
end
Base.length(K::ProductCone) = sum(length.(K.cones))

function project!(x::AbstractVector{T}, K::ProductCone, y::AbstractVector{T}) where {T}
    ind = 1
    for (i, Ki) in enumerate(K.cones)
        inds = ind:(ind + length(Ki) - 1) 
        @views project!(x[inds], Ki, y[inds])
        ind += length(Ki)
    end
    return nothing
end

function support(x::AbstractVector{T}, K::ProductCone) where {T}
    ind = 1
    ret = zero(T)
    for (i, Ki) in enumerate(K.cones)
        inds = ind:(ind + length(Ki) - 1) 
        @views ret += support(x[inds], Ki)
        ind += length(Ki)
    end
    return ret
end

function in_recession_cone(x::AbstractVector{T}, K::ProductCone, tol::T) where {T}
    ind = 1
    for (i, Ki) in enumerate(K.cones)
        inds = ind:(ind + length(Ki) - 1) 
       @views !in_recession_cone(x[inds], Ki, tol) && return false
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

Base.length(K::IntervalCone) = Base.length(K.l)

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


# Code below is experimental and not documented or well tested
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exponential Cone = {(x,y,z) | y > 0, y*exp(x/y) ≤ z}
struct ExponentialCone <: Cone end
Base.length(K::ExponentialCone) = 3

# code from https://github.com/matbesancon/MathOptSetDistances.jl
# we project onto the closure:
#   cl(K) = {(x,y,z) | y > 0, y*exp(x/y) ≤ z} ∪ {(x,y,z) | x ≤ 0, y = 0, z ≥ 0}
function project!(x::AbstractVector{T}, K::ExponentialCone, y::AbstractVector{T}) where {T}
    
    if in_cone(y, K)
        x .= y
    elseif in_polar_cone(y, K)
        # if in polar cone Ko = -K^*
        fill!(x, zero(T))
    elseif y[1] <= 0 && y[2] <= 0
        x[1] = y[1]
        x[2] = zero(T)
        x[3] = max(y[3], zero(T))
    else
        _exp_cone_proj_case_4!(x, y)
    end

    return nothing
end

function in_cone(v::AbstractVector{T}, ::ExponentialCone, tol=1e-8) where {T}
    return (
        (v[1] <= zero(T) && isapprox(v[2], zero(T), atol=tol) && v[3] >= zero(T)) ||
        (v[2] > zero(T) && v[2] * exp(v[1] / v[2]) - v[3] <= tol)
    )
end

function in_polar_cone(v::AbstractVector{T}, ::ExponentialCone; tol=1e-8) where {T}
    return (
            (isapprox(v[1], 0, atol=tol) && -v[2] >= 0 && -v[3] >= 0) ||
            (-v[1] < 0 && -v[1]*exp(v[2]/v[1]) + ℯ * -v[3] >= tol)
        )
end

function _exp_cone_proj_case_4!(u::AbstractVector{T}, v::AbstractVector{T}; tol=1e-8) where {T}
    # Try Heuristic solution [Friberg 2021, Lemma 5.1]
    # vp = proj onto primal cone, vd = proj onto polar cone
    vp = SVector{3,T}(min(v[1], 0), zero(T), max(v[3], 0))
    vd = SVector{3,T}(zero(T), min(v[2], 0), min(v[3], 0))
    
    if v[2] > 0
        zp = max(v[3], v[2]*exp(v[1]/v[2]))
        if zp - v[3] < norm(vp - v)
            vp = SVector{3,T}(v[1], v[2], zp)
        end
    end
    if v[1] > 0
        zd = min(v[3], -v[1]*exp(v[2]/v[1] - 1))
        if v[3] - zd < norm(vd - v)
            vd = SVector{3,T}(v[1], v[2], zd)
        end
    end

    # Check if heuristics above approximately satisfy the optimality conditions
    # Friberg 2021 eq (6)
    v3 = SVector{3,T}(vp[1] + vd[1] - v[1], vp[2] + vd[2] - v[2], vp[3] + vd[3] - v[3])
    opt_norm = norm(v3)
    opt_ortho = abs(dot(vp, vd))
    if norm(v - vp) < tol || norm(v - vd) < tol || (opt_norm < tol && opt_ortho < tol)
        u .= vp
    end

    # Failure of heuristics -> non heuristic solution
    # Ref: https://docs.mosek.com/slides/2018/ismp2018/ismp-friberg.pdf, p47-48
    # Thm: h(x) is smooth, strictly increasing, and changes sign on domain
    r, s, t = v[1], v[2], v[3]
    @inline h(x) = (((x-1)*r + s) * exp(x) - (r - x*s)*exp(-x))/(x^2 - x + 1) - t

    # Note: won't both be Inf by case 3 of projection
    lb = r > 0 ? 1 - s/r : -Inf
    ub = s > 0 ? r/s : Inf

    # Deal with ±Inf bounds
    if isinf(lb)
        lb = min(ub-0.125, -0.125)
        for _ in 1:10
            h(lb) < 0 && break
            ub = lb
            lb *= 2
        end
    end
    if isinf(ub)
        ub = max(lb+0.125, 0.125)
        for _ in 1:10
            h(ub) > 0 && break
            lb = ub
            ub *= 2
        end
    end

    # Check bounds
    if !(h(lb) < 0 && h(ub) > 0)
        error("Failure to find bracketing interval for exp cone projection.")
    end

    x = _bisection(h, lb, ub)
    if x === nothing
        error("Failure in root-finding for exp cone projection with boundaries ($lb, $ub).")
    end

    u .= ((x - 1) * r + s)/(x^2 - x + 1) * SVector{3,T}(x, 1, exp(x))
end

# For any convex cone:
# supp(x) = sup {xᵀy | y ∈ K} = 0 if x ∈ Kᵒ, Inf otherwise
function support(x::AbstractVector{T}, K::ExponentialCone) where {T}
    if in_polar_cone(x, K)
        return zero(T)
    else
        return Inf
    end
end

function in_recession_cone(x::AbstractVector{T}, K::ExponentialCone, tol::T) where {T}
    return in_cone(x, K, tol)
end

function _bisection(f, left, right; max_iters=1000, tol=1e-14)
    # STOP CODES:
    #   0: Success (floating point limit or exactly 0)
    #   1: Failure (max_iters without coming within tolerance of 0)
    for _ in 1:max_iters
        f_left, f_right = f(left), f(right)
        sign(f_left) == sign(f_right) && error("
            Interval became non-bracketing.
            \nL: f($left) = $f_left
            \nR: f($right) = $f_right"
        )

        # Terminate if interval length ~ eps()
        mid = (left + right) / 2
        if left == mid || right == mid
            return mid
        end
        # Terminate if within tol of 0; otherwise, bisect
        f_mid = f(mid)
        if abs(f_mid) < tol
            return mid
        end
        if signbit(f_mid) == signbit(f_left)
            left = mid
        elseif signbit(f_mid) == signbit(f_right)
            right = mid
        end
    end
    return nothing
end
