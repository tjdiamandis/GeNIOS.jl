abstract type Solver end
abstract type ProblemData end
abstract type OptimizerCache end

function Base.show(io::IO, solver::Solver)
    print(io, "A $(typeof(solver).name.name) GeNIOS Solver\n")
    n = solver.data.n
    m = solver.data.m
    print(io, "  num vars x:  $n\n")
    print(io, "  num vars z:  $m\n")
    t_solver = typeof(solver)
    if t_solver <: MLSolver
        print(io, "  samples:     $(solver.data.N)\n")
    else
        print(io, "  constraints: $m\n")
    end
end

struct MLProblemData{
    T <: Real,
    V <: AbstractVector{T}, 
    M <: AbstractMatrix{T},
} <: ProblemData
    M
    c
    Adata::M
    bdata::V
    N::Int
    m::Int
    n::Int
    d2f::Function
    df::Function
    f::Function
    fconj::Function
end

struct ConicProgramData{T} <: ProblemData
    M
    c
    m::Int
    n::Int
    P
    q::AbstractVector{T}
    K::Cone
end

struct GenericProblemData{T, ParamType} <: ProblemData
    M
    c::AbstractVector{T}
    m::Int
    n::Int
    Hf::HessianOperator
    grad_f!::Function
    f::Function
    g::Function
    prox_g!::Function
    params::ParamType
end

@kwdef struct GenericCache{T} <: OptimizerCache
    vm::Vector{T}
    vn::Vector{T}
    vn2::Vector{T}
    rhs::Vector{T}
end

function init_cache(data::GenericProblemData{T}) where {T <: Real}
    m, n = data.m, data.n
    return GenericCache(
        vm=zeros(T, m),
        vn=zeros(T, n),
        vn2=zeros(T, n),
        rhs=zeros(T, n),
    )
end

@kwdef struct ConicCache{T} <: OptimizerCache
    vm::Vector{T}
    vn::Vector{T}
    vn2::Vector{T}
    rhs::Vector{T}
end

function init_cache(data::ConicProgramData{T}) where {T <: Real}
    m, n = data.m, data.n
    return ConicCache(
        vm=zeros(T, m),
        vn=zeros(T, n),
        vn2=zeros(T, n),
        rhs=zeros(T, n),
    )
end

@kwdef struct MLCache{T} <: OptimizerCache
    vm::Vector{T}
    vn::Vector{T}
    vN::Vector{T}
    vn2::Vector{T}
    rhs::Vector{T}
end

function init_cache(data::MLProblemData{T}) where {T <: Real}
    m, n, N = data.m, data.n, data.N
    return MLCache(
        vm=zeros(T, m),
        vn=zeros(T, n),
        vN=zeros(T, N),
        vn2=zeros(T, n),
        rhs=zeros(T, n),
    )
end

# TODO: maybe make immutable?
mutable struct GenericSolver{
    T <: Real,
    V <: AbstractVector{T}
} <: Solver
    data::GenericProblemData{T} # data
    lhs_op::LinearOperator{T}   # LinerOperator for LHS of x update system
    P::Union{UniformScaling{Bool}, NystromPreconditionerInverse{T}} # preconditioner
    xk::V                       # var   : primal (loss)
    Mxk::V                      # var   : primal 
    zk::V                       # var   : primal (reg)
    zk_old::V                   # var   : dual (prev step)
    uk::V                       # var   : dual
    rp::V                       # resid : primal
    rd::V                       # resid : dual
    rp_norm::T                  # resid_norm : primal
    rd_norm::T                  # resid_norm : dual
    obj_val::T                  # log   : f(x) + g(z)
    loss::T                     # log   : TODO: rem
    dual_gap::T                 # log   : TODO: rem
    ρ::T                        # param : ADMM penalty
    cache::GenericCache{T}      # cache : cache for intermediate results
end
function GenericSolver(
    f, 
    grad_f!, 
    Hf, 
    g, 
    prox_g!, 
    M, 
    c::Vector{T}; 
    params=nothing
) where {T}
    m, n = typeof(M) <: UniformScaling ? (length(c), length(c)) : size(M)
    data = GenericProblemData(M, c, m, n, Hf, grad_f!, f, g, prox_g!, params)
    xk = zeros(T, n)
    Mxk = zeros(T, m)
    zk = zeros(T, m)
    zk_old = zeros(T, m)
    uk = zeros(T, m)
    rp = zeros(T, m)
    rd = zeros(T, n)
    obj_val, loss, dual_gap = zero(T), zero(T), zero(T)
    rp_norm, rd_norm = zero(T), zero(T)
    ρ = one(T)
    cache = init_cache(data)
    
    return GenericSolver(
        data, 
        LinearOperator(M, one(T), Hf, m, n), 
        I,
        xk, Mxk, zk, zk_old, uk, rp, rd, 
        rp_norm, rd_norm,
        obj_val, loss, dual_gap,
        ρ,
        cache)
end

function GenericSolver(data::ProblemData)
    return GenericSolver(
        data.f,
        data.grad_f!,
        data.Hf,
        data.g,
        data.prox_g!,
        data.M,
        data.c
    )
end

#TODO: split inequality and equality constraints??
# TODO: should P be in lhs_op?
mutable struct ConicSolver{
    T <: Real,
    V <: AbstractVector{T},
} <: Solver
    data::ConicProgramData{T} # data
    lhs_op::LinearOperator{T}   # LinerOperator for LHS of x update system
    P::Union{UniformScaling{Bool}, NystromPreconditionerInverse{T}} # preconditioner
    xk::V                       # var   : primal (loss)
    δx::V                       # var   : primal  
    Mxk::V                      # var   : primal 
    zk::V                       # var   : primal (reg)
    zk_old::V                   # var   : dual (prev step)
    uk::V                       # var   : dual
    δy::V                       # var   : dual
    rp::V                       # resid : primal
    rd::V                       # resid : dual
    rp_norm::T                  # resid_norm : primal
    rd_norm::T                  # resid_norm : dual
    obj_val::T                  # log   : f(x) + g(z)
    loss::T                     # log   : TODO: rem
    dual_gap::T                 # log   : TODO: rem
    ρ::T                        # param : ADMM penalty
    cache::ConicCache{T}        # cache : cache for intermediate results
end
function ConicSolver(
    P, 
    q, 
    K, 
    M, 
    c::Vector{T}; 
    σ=nothing, 
    check_dims=true
) where {T}
    check_dims && check_data_dims(P, q, K, M, c) # errors if mismatch
    m, n = typeof(M) <: UniformScaling ? (length(c), length(c)) : size(M)
    data = ConicProgramData(M, c, m, n, P, q, K)
    xk = zeros(T, n)
    Mxk = zeros(T, m)
    δx = zeros(T, n)
    zk = zeros(T, m)
    zk_old = zeros(T, m)
    uk = zeros(T, m)
    δy = zeros(T, m)
    rp = zeros(T, m)
    rd = zeros(T, n)
    obj_val, loss, dual_gap = zero(T), zero(T), zero(T)
    rp_norm, rd_norm = zero(T), zero(T)
    ρ = one(T)
    cache = init_cache(data)

    σ = isnothing(σ) ? 1e-6 : σ
    
    return ConicSolver(
        data, 
        LinearOperator(M, one(T), ConicHessianOperator(P, σ), m, n), 
        I,
        xk, δx, Mxk, zk, zk_old, uk, δy, rp, rd, 
        rp_norm, rd_norm,
        obj_val, loss, dual_gap,
        ρ,
        cache
    )
end

function check_data_dims(P, q, K, M, c)
    m, n = typeof(M) <: UniformScaling ? (length(c), length(c)) : size(M)
    size_P = typeof(P) <: UniformScaling ? (length(q), length(q)) : size(P)
    if size_P[1] != n || size_P[2] != n || length(q) != n
        error("objective data dimensions do not match constraint matrix M")
    elseif length(K) != m || length(c) != m
        error("constraint data dimensions do not match constraint matrix M")
    end

    return nothing
end

function QPSolver(P, q, M, l, u; σ=nothing, check_dims=true)
    m =  typeof(M) <: UniformScaling ? length(q) : size(M, 1)
    # TODO: Optimal QP step size: https://www.merl.com/publications/docs/TR2014-050.pdf
    # - perhaps estimate with randomized method??
    # - strictly convex case
    # ρ = sqrt(λmin(P)*λmax(P))
    return ConicSolver(P, q, IntervalCone(l, u), M, zeros(m); σ=σ, check_dims=check_dims)
end

mutable struct MLSolver{
    T <: Real,
    V <: AbstractVector{T}, 
    M <: AbstractMatrix{T},
} <: Solver
    data::MLProblemData{T, V, M}# data
    lhs_op::LinearOperator{T}   # LinerOperator for LHS of x update system
    P::Union{UniformScaling{Bool}, NystromPreconditionerInverse{T}} # preconditioner
    xk::V                       # var   : primal
    Mxk::V                      # var   : primal
    pred::V                     # var   : primal (Adata*zk - bdata) TODO: zk vs xk?
    zk::V                       # var   : primal (reg)
    zk_old::V                   # var   : dual (prev step)
    uk::V                       # var   : dual
    rp::V                       # resid : primal
    rd::V                       # resid : dual
    rp_norm::T                  # resid_norm : primal
    rd_norm::T                  # resid_norm : dual
    obj_val::T                  # log   : 0.5*∑f(aᵢᵀx - bᵢ) + λ₁|x|₁ + λ₂/2|x|₂²
    loss::T                     # log   : 0.5*∑f(aᵢᵀx - bᵢ) (uses zk)
    dual_gap::T                 # log   : obj_val - g(ν)
    ρ::T                        # param : ADMM penalty
    λ1::T                       # param : l1 regularization
    λ2::T                       # param : l2 regularization
    cache::MLCache{T}           # cache : cache for intermediate results
end

function MLSolver(f,
    df,
    d2f,
    λ1,
    λ2,
    Adata::M,
    bdata::V; 
    fconj=()->error(ArgumentError("dual gap used but fconj not defined"))
) where {T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    N, n = size(Adata)
    #TODO: may want to add offset?
    m = n

    data = MLProblemData(I, zeros(T, n), Adata, bdata, N, m, n, d2f, df, f, fconj)
    xk = zeros(T, n)
    Mxk = zeros(T, n)
    pred = zeros(T, N)
    zk = zeros(T, m)
    zk_old = zeros(T, n)
    uk = zeros(T, n)
    rp = zeros(T, n)
    rd = zeros(T, n)
    obj_val, loss, dual_gap = zero(T), zero(T), zero(T)
    rp_norm, rd_norm = zero(T), zero(T)
    ρ = one(T)
    cache = init_cache(data)

    Hf = MLHessianOperator(Adata, bdata, λ2) 

    return MLSolver(
        data, 
        LinearOperator(I, one(T), Hf, m, n),
        I,
        xk, Mxk, pred, zk, zk_old, uk, rp, rd, 
        rp_norm, rd_norm,
        obj_val, loss, dual_gap,
        ρ,
        λ1, λ2,
        cache
    )
end

# Fills in first and second derivative if not defined
function MLSolver(
    f,
    λ1::S,
    λ2::S,
    Adata::AbstractMatrix{T},
    bdata::AbstractVector{T}; 
    fconj=x->error(ArgumentError("dual gap used but fconj not defined"))
) where {T <: Real, S <: Number}
    λ1 = convert(T, λ1)
    λ2 = convert(T, λ2)
    df = x->derivative(f, x)
    d2f = x->derivative(y->derivative(f, y), x)
    return MLSolver(f, df, d2f, λ1, λ2, Adata, bdata, fconj=fconj)
end

function LassoSolver(
    λ1::S,
    Adata::AbstractMatrix{T},
    bdata::AbstractVector{T}
) where {T <: Real, S <: Number}
    λ1 = convert(T, λ1)
    λ2 = zero(T)
    f(x) = 0.5*x^2 
    df(x) = x
    d2f(x) = 1.0
    fconj(x) = 0.5*x^2
    return MLSolver(f, df, d2f, λ1, λ2, Adata, bdata, fconj=fconj)
end

function ElasticNetSolver(
    λ1::S,
    λ2::S,
    Adata::AbstractMatrix{T},
    bdata::AbstractVector{T}
) where {T <: Real, S <: Number}
    λ1 = convert(T, λ1)
    λ2 = convert(T, λ2)
    f(x) = 0.5*x^2 
    df(x) = x
    d2f(x) = 1.0
    fconj(x) = 0.5*x^2
    return MLSolver(f, df, d2f, λ1, λ2, Adata, bdata, fconj=fconj)
end

function LogisticSolver(
    λ1::S,
    λ2::S,
    Adata::AbstractMatrix{T},
    bdata::AbstractVector{T}
) where {T <: Real, S <: Number}
    λ1 = convert(T, λ1)
    λ2 = convert(T, λ2)
    f(x) = GeNIOS.log1pexp(x)
    df(x) = GeNIOS.logistic(x)
    d2f(x) = GeNIOS.logistic(x) / GeNIOS.log1pexp(x)
    fconj(x::T) where {T} = x ≥ 0 && x ≤ 1 ? (one(T) - x) * log(one(T) - x) + x * log(x) : Inf
    return MLSolver(f, df, d2f, λ1, λ2, Adata, bdata, fconj=fconj)
end


@kwdef mutable struct SolverOptions{T <: AbstractFloat, S <: Real}
    ρ0::T = 1.0                                     # param : ADMM penalty
    α::T = 1.6                                      # param : relaxation
    relax::Bool = true
    logging::Bool = true
    precondition::Bool = false
    dual_gap_tol::T = 1e-4
    relax_tol::T = 1e-3
    max_iters::Int = 4000
    max_time_sec::T = 1200.0
    print_iter::Int = 25
    rho_update_iter::Int = 50
    sketch_update_iter::Int = 20
    verbose::Bool = true
    linsys_max_tol::T = 1.0
    eps_abs::T = 1e-4
    eps_rel::T = 1e-4
    eps_inf::T = 1e-8
    norm_type::S = 2
    use_dual_gap::Bool = false
    update_preconditioner::Bool = true
    infeas_check_iter::Int = 25
    num_threads::Int = Sys.CPU_THREADS
    init_sketch_size::Int = 50                      # param : preconditioner rank
    use_adaptive_sketch::Bool = false
    adaptive_sketch_tol::Float64 = eps()
    linsys_exponent::T = 1.2
end

function Base.show(io::IO, options::SolverOptions)
    print(io, "GeNIOS SolverOptions:\n")
    for property in propertynames(options)
        @printf(io, "    %-20s: %s\n", string(property), getfield(options, property))
    end
end


struct GeNIOSLog{T <: AbstractFloat, S <: AbstractVector{T}}
    dual_gap::Union{S, Nothing}
    obj_val::Union{S, Nothing}
    iter_time::Union{S, Nothing}
    linsys_time::Union{S, Nothing}
    prox_time::Union{S, Nothing}
    rp::Union{S, Nothing}
    rd::Union{S, Nothing}
    setup_time::T
    precond_time::T
    solve_time::T
end
function GeNIOSLog(setup_time::T, precond_time::T, solve_time::T) where {T <: AbstractFloat}
    return GeNIOSLog(
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        setup_time, precond_time, solve_time
    )
end

function Base.show(io::IO, log::GeNIOSLog)
    print(io, "--- GeNIOSLog ---\n")
    !isnothing(log.rp) && print(io, "num iters:  $(length(log.rp))\n")
    print(io, "setup time:  $(round(log.setup_time, digits=3))s\n")
    print(io, " - pc time:  $(round(log.precond_time, digits=3))s\n")
    print(io, "solve time:  $(round(log.solve_time, digits=3))s\n")
end

function create_temp_log(solver::Solver, max_iters::Int)
    T = eltype(solver.xk)
    return GeNIOSLog(
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zero(T),
        zero(T),
        zero(T)
    )
end

# Status is one of:
# - OPTIMAL
# - INFEASIBLE
# - DUAL_INFEASIBLE
# - ITERATION_LIMIT
# - TIME_LIMIT
struct GeNIOSResult{T}
    status::Symbol
    obj_val::T                 # primal objective
    loss::T                    # TODO:
    x::AbstractVector{T}       # primal soln
    z::AbstractVector{T}       # primal soln
    u::AbstractVector{T}       # dual soln
    ρ::T                       # ADMM penalty
    dual_gap::T                # duality gap
    log::GeNIOSLog{T}
end

function Base.show(io::IO, result::GeNIOSResult)
    print(io, "--- GeNIOSResult ---\n")
    print(io, "Status:     ")
    color = result.status == :OPTIMAL ? :green : :red
    printstyled(io, " $(result.status)\n", color=color)
    print(io, "Obj. val:   $(@sprintf("%.4g", result.obj_val))\n")
    !isnothing(result.log.rp) && print(io, "num iters:  $(length(result.log.rp))\n")
    print(io, "setup time: $(round(result.log.setup_time, digits=3))s\n")
    print(io, "solve time: $(round(result.log.solve_time, digits=3))s\n")
end

function populate_log!(genios_log, solver::Solver, ::SolverOptions, t, time_sec, time_linsys, time_prox)
    genios_log.obj_val[t] = solver.obj_val
    genios_log.iter_time[t] = time_sec
    genios_log.linsys_time[t] = time_linsys
    genios_log.prox_time[t] = time_prox
    genios_log.rp[t] = solver.rp_norm
    genios_log.rd[t] = solver.rd_norm
    return nothing
end

function populate_log!(genios_log, solver::MLSolver, options::SolverOptions, t, time_sec, time_linsys, time_prox)
    genios_log.obj_val[t] = solver.obj_val
    genios_log.iter_time[t] = time_sec
    genios_log.linsys_time[t] = time_linsys
    genios_log.prox_time[t] = time_prox
    genios_log.rp[t] = solver.rp_norm
    genios_log.rd[t] = solver.rd_norm

    if options.use_dual_gap
        genios_log.dual_gap[t] = solver.dual_gap
    end
    return nothing
end