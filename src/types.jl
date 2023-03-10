
struct ProblemData{T}
    A
    c::AbstractVector{T}
    m::Int
    n::Int
    Hf::HessianOperator
    grad_f!::Function
    f::Function
    g::Function
    prox_g!::Function
end

# TODO: maybe make immutable?
mutable struct Solver{T}
    data::ProblemData{T}        # data
    lhs_op::LinearOperator{T}   # LinerOperator for LHS of x update system
    P                           # preconditioner; TODO: combine with lhs_op?
    xk::AbstractVector{T}       # var   : primal (loss)
    Axk::AbstractVector{T}      # var   : primal 
    zk::AbstractVector{T}       # var   : primal (reg)
    zk_old::AbstractVector{T}   # var   : dual (prev step)
    uk::AbstractVector{T}       # var   : dual
    rp::AbstractVector{T}       # resid : primal
    rd::AbstractVector{T}       # resid : dual
    rp_norm::T                  # resid_norm : primal
    rd_norm::T                  # resid_norm : dual
    obj_val::T                  # log   : 0.5*||Ax - b||² + γ|x|₁
    loss::T                     # log   : 0.5*||Ax - b||² (uses zk)
    dual_gap::T                 # log   : obj_val - g(ν)
    ρ::T                        # param : ADMM penalty
    α::T                        # param : relaxation
    cache                       # cache : cache for intermediate results
end
function Solver(f, grad_f!, Hf, g, prox_g!, A, c::AbstractVector{T}; ρ=1.0, α=1.0) where {T}
    @show A
    m, n = A == I ? (length(c), length(c)) : size(A)
    data = ProblemData(A, c, m, n, Hf, grad_f!, f, g, prox_g!)
    xk = zeros(T, n)
    Axk = zeros(T, m)
    zk = zeros(T, m)
    zk_old = zeros(T, m)
    uk = zeros(T, m)
    rp = zeros(T, m)
    rd = zeros(T, n)
    obj_val, loss, dual_gap = zero(T), zero(T), zero(T)
    rp_norm, rd_norm = zero(T), zero(T)
    cache = init_cache(data)
    
    return Solver(
        data, 
        LinearOperator(A, ρ, Hf, m, n), 
        I,
        xk, Axk, zk, zk_old, uk, rp, rd, 
        rp_norm, rd_norm,
        obj_val, loss, dual_gap,
        ρ, α,
        cache)
end

function Solver(data::ProblemData; ρ=1.0, α=1.0)
    return Solver(
        data.f,
        data.grad_f!,
        data.Hf,
        data.g,
        data.prox_g!,
        data.A,
        data.c,
        ρ=ρ,
        α=α
    )
end

function init_cache(data::ProblemData{T}) where {T <: Real}
    m, n = data.m, data.n
    return (
        vm=zeros(T, m),
        vn=zeros(T, n),
        vn2=zeros(T, n),
        rhs=zeros(T, n),
    )
end

struct SolverOptions
    relax::Bool
    logging::Bool
    indirect::Bool
    precondition::Bool
    tol::Real
    max_iters::Int
    max_time_sec::Real
    print_iter::Int
    rho_update_iter::Int
    sketch_update_iter::Int
    verbose::Bool
    multithreaded::Bool
    linsys_max_tol::Real
    eps_abs::Real
    eps_rel::Real
    norm_type::Real
end

function SolverOptions(;
    relax=true,
    logging=true,
    indirect=false,
    precondition=true,
    tol=1e-4,
    max_iters=100,
    max_time_sec=1200.0,
    print_iter=1,
    rho_update_iter=50,
    sketch_update_iter=20,
    verbose=true,
    multithreaded=false,
    linsys_max_tol=1e-1,
    eps_abs=1e-4,
    eps_rel=1e-4,
    norm_type=2,
)
    return SolverOptions(
        relax,
        logging,
        indirect,
        precondition,
        tol,
        max_iters,
        max_time_sec,
        print_iter,
        rho_update_iter,
        sketch_update_iter,
        verbose,
        multithreaded,
        linsys_max_tol,
        eps_abs,
        eps_rel,
        norm_type
    )
end


struct NysADMMLog{T <: AbstractFloat, S <: AbstractVector{T}}
    dual_gap::Union{S, Nothing}
    obj_val::Union{S, Nothing}
    iter_time::Union{S, Nothing}
    linsys_time::Union{S, Nothing}
    rp::Union{S, Nothing}
    rd::Union{S, Nothing}
    setup_time::T
    precond_time::T
    solve_time::T
end
function NysADMMLog(setup_time::T, precond_time::T, solve_time::T) where {T <: AbstractFloat}
    return NysADMMLog(
        nothing, nothing, nothing, nothing, nothing, nothing,
        setup_time, precond_time, solve_time
    )
end

function create_temp_log(::Solver{T}, max_iters::Int) where {T}
    return NysADMMLog(
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

struct NysADMMResult{T}
    obj_val::T                 # primal objective
    loss::T                # 0.5*||Ax - b||²
    x::AbstractVector{T}       # primal soln
    z::AbstractVector{T}       # primal soln
    dual_gap::T                # duality gap
    # vT::AbstractVector{T}      # dual certificate
    log::NysADMMLog{T}
end