function build_preconditioner!(solver::Solver, options::SolverOptions)
    !options.precondition && return 0.0
    options.init_sketch_size = solver.data.n ≥ 1_000 ? options.init_sketch_size : solver.data.n ÷ 20
    precond_time_start = time_ns()
    _build_preconditioner!(solver, options)
    return (time_ns() - precond_time_start) / 1e9
end

# Conic solver currently defaults here
# TODO: possibly want to include ρMᵀM in the preconditioner??
# - would need to estimate the smallest eigenvalue for the regularization, I think
# - alternatively, could use the OSQP approach
function _build_preconditioner!(solver::Solver, options::SolverOptions)
    update!(solver.lhs_op.Hf_xk, solver)
    if options.use_adaptive_sketch
        ∇²fx_nys = adaptive_nystrom_sketch(
            solver.lhs_op.Hf_xk, solver.lhs_op.n,
            options.init_sketch_size;
            tol=options.adaptive_sketch_tol*solver.data.n^2
        )
    else    
        ∇²fx_nys = RP.NystromSketch(solver.lhs_op.Hf_xk, options.init_sketch_size; n=solver.lhs_op.n)
    end
    solver.P = RP.NystromPreconditionerInverse(∇²fx_nys, solver.ρ)
    return nothing
end

function _build_preconditioner!(solver::MLSolver, options::SolverOptions)
    update!(solver.lhs_op.Hf_xk, solver)
    if options.use_adaptive_sketch
        ∇²fx_nys = adaptive_nystrom_sketch(
            solver.lhs_op.Hf_xk, solver.lhs_op.n,
            options.init_sketch_size;
            tol=options.adaptive_sketch_tol*solver.data.n^2
        )
    else    
        ∇²fx_nys = RP.NystromSketch(solver.lhs_op.Hf_xk, options.init_sketch_size; n=solver.lhs_op.n)
    end
    
    # Adds the ℓ2 regularization term to the preconditioner
    solver.P = RP.NystromPreconditionerInverse(∇²fx_nys, solver.ρ + solver.λ2)
    return nothing
end

# TODO: which sketch size should this be?
function update_preconditioner!(solver::Solver, options::SolverOptions)
    !options.update_preconditioner && return nothing
    update!(solver.lhs_op.Hf_xk, solver)
    ∇²fx_nys = RP.NystromSketch(solver.lhs_op.Hf_xk, options.init_sketch_size; n=solver.lhs_op.n)
    solver.P = RP.NystromPreconditionerInverse(∇²fx_nys, solver.ρ)
    return nothing
end

function update_preconditioner!(solver::MLSolver, options::SolverOptions)
    !options.update_preconditioner && return nothing
    update!(solver.lhs_op.Hf_xk, solver)
    ∇²fx_nys = RP.NystromSketch(solver.lhs_op.Hf_xk, options.init_sketch_size; n=solver.lhs_op.n)

    # Adds the ℓ2 regularization term to the preconditioner
    solver.P = RP.NystromPreconditionerInverse(∇²fx_nys, solver.ρ + solver.λ2)
    return nothing
end

function update_preconditioner_rho!(solver::Solver, options::SolverOptions)
    solver.P = RP.NystromPreconditionerInverse(solver.P.A_nys, solver.ρ)
    return nothing
end

function update_preconditioner_rho!(solver::MLSolver, options::SolverOptions)
    solver.P = RP.NystromPreconditionerInverse(solver.P.A_nys, solver.ρ + solver.λ2)
    return nothing
end

function update_rho!(solver::Solver, options::SolverOptions)
    # TODO:
    # https://proceedings.mlr.press/v54/xu17a/xu17a.pdf

    μ = 10
    τ = 2
    if solver.rp_norm > μ * solver.rd_norm
        solver.ρ = solver.ρ * τ
        solver.lhs_op.ρ[1] = solver.ρ
        @. solver.uk *= 1 / τ
        return true
    elseif solver.rd_norm > μ * solver.rp_norm
        solver.ρ = solver.ρ / τ
        solver.lhs_op.ρ[1] = solver.ρ
        @. solver.uk *= τ
        return true
    end

    return false
end

function converged(solver::Solver, options::SolverOptions, backup=true)
    mul!(solver.cache.vm, solver.data.M, solver.xk)
    norm_Ax = norm(solver.cache.vm)
    norm_z = norm(solver.zk)
    norm_c = norm(solver.data.c)
    eps_pri = sqrt(solver.data.m) * options.eps_abs + options.eps_rel * max(norm_Ax, norm_z, norm_c)

    mul!(solver.cache.vn, solver.data.M', solver.uk)
    norm_Aty = norm(solver.cache.vn)
    eps_dual = sqrt(solver.data.n) * options.eps_abs + options.eps_rel * norm_Aty

    return solver.rp_norm ≤ eps_pri && solver.rd_norm ≤ eps_dual
end

function converged(solver::MLSolver, options::SolverOptions)
    if options.use_dual_gap
        return solver.dual_gap ≤ options.dual_gap_tol
    end

    return converged(solver, options, true)
end

# TODO: switch to zk for f?????
function obj_val!(solver::Solver, options::SolverOptions)
    solver.obj_val = solver.data.f(solver.xk) + solver.data.g(solver.zk)
end

# TODO: switch to zk for f?????
function obj_val!(solver::MLSolver, options::SolverOptions)
    # pred = Ax - b
    mul!(solver.pred, solver.data.Adata, solver.zk)
    solver.pred .-= solver.data.bdata
    
    solver.loss = sum(x->solver.data.f(x), solver.pred)
    solver.obj_val = solver.loss + 
        solver.λ1*norm(solver.zk, 1) + (solver.λ2/2)*sum(abs2, solver.zk)
end

function obj_val!(solver::ConicSolver, options::SolverOptions)
    mul!(solver.cache.vn, solver.data.P, solver.xk)
    solver.obj_val = 0.5*dot(solver.xk, solver.cache.vn) + dot(solver.data.q, solver.xk)
end

# Used to implement custom convergence criteria (e.g., via dual gap)
function convergence_criteria!(::Solver, ::SolverOptions)
    return nothing
end

# TODO: opportunities for optimization for logistic and lasso
function convergence_criteria!(solver::MLSolver, options::SolverOptions)
    !options.use_dual_gap && return nothing

    ν = solver.cache.vN
    ν .= solver.data.df.(solver.pred)
    mul!(solver.cache.vn, solver.data.Adata', ν)
    @. solver.cache.vn += solver.λ2 * solver.zk
    
    # Applies normalization
    normalization = solver.λ1 / norm(solver.cache.vn, Inf)
    ν .*= normalization

    # g(ν) = -∑f*(νᵢ) - bᵀνᵢ
    dual_obj = -sum(x->solver.data.fconj(x), ν)
    dual_obj -= dot(solver.data.bdata, ν)

    # Adds extra term to g(ν), -(1/2λ₂)∑( (|Aᵀν| - λ₁)₊ )², for λ₂ > 0
    if !iszero(solver.λ2)
        mul!(solver.cache.vn, solver.data.Adata', ν)
        @. solver.cache.vn = abs(solver.cache.vn)
        solver.cache.vn .-= solver.λ1
        @. solver.cache.vn = (max(solver.cache.vn, zero(eltype(solver.cache.vn))))^2
        dual_obj -= 1/2solver.λ2 * sum(solver.cache.vn)
    end

    solver.dual_gap = (solver.obj_val - dual_obj) / max(abs(solver.obj_val), abs(dual_obj))
    return nothing
end

function compute_rhs!(solver::Solver, options::SolverOptions)
    # RHS = ∇²f(xᵏ)xᵏ - ∇f(xᵏ) + ρAᵀ(zᵏ + c - uᵏ)
    mul!(solver.cache.vn, solver.data.Hf, solver.xk)
    solver.data.grad_f!(solver.cache.vn2, solver.xk)
    
    @. solver.cache.vm = solver.zk + solver.data.c - solver.uk
    mul!(solver.cache.rhs, solver.data.M', solver.cache.vm)
    @. solver.cache.rhs = solver.cache.vn - solver.cache.vn2 + solver.ρ * solver.cache.rhs
    
    return nothing
end

# NOTE: Assumes that pred has been computed
function compute_rhs!(solver::MLSolver, options::SolverOptions)
    # RHS = ∇²f(xᵏ)xᵏ - ∇f(xᵏ) + ρAᵀ(zᵏ + c - uᵏ)

    # compute first term (hessian)
    mul!(solver.cache.vN, solver.data.Adata, solver.xk)
    @. solver.cache.vN *= solver.data.d2f(solver.pred)
    mul!(solver.cache.vn, solver.data.Adata', solver.cache.vN)

    # compute second term (gradient)
    @. solver.cache.vN = solver.data.df(solver.pred)
    mul!(solver.cache.vn2, solver.data.Adata', solver.cache.vN)
    
    # compute last term
    @. solver.cache.vm = solver.zk + solver.data.c - solver.uk
    mul!(solver.cache.rhs, solver.data.M', solver.cache.vm)

    # add them up
    @. solver.cache.rhs = solver.cache.vn - solver.cache.vn2 + solver.ρ * solver.cache.rhs
    
    return nothing
end

function compute_rhs!(solver::ConicSolver, options::SolverOptions)
    # RHS = ∇²f(xᵏ)xᵏ - ∇f(xᵏ) + ρAᵀ(zᵏ + c - uᵏ) = -q + ρAᵀ(zᵏ + c - uᵏ)
    @. solver.cache.vn = -solver.data.q
    
    @. solver.cache.vm = solver.zk + solver.data.c - solver.uk
    mul!(solver.cache.rhs, solver.data.M', solver.cache.vm)
    @. solver.cache.rhs = solver.cache.vn + solver.ρ * solver.cache.rhs
    
    return nothing
end

function update_δx!(::AbstractVector, ::Solver, ::SolverOptions)
    return nothing
end

function update_δx!(x::AbstractVector, solver::ConicSolver, ::SolverOptions)
    @. solver.δx = x - solver.xk
    return nothing
end

function update_x!(
    solver::Solver,
    options::SolverOptions,
    linsys_solver::CgSolver
)
    # RHS = ∇²f(xᵏ)xᵏ - ∇f(xᵏ) - ρAᵀ(zᵏ - c + uᵏ)
    compute_rhs!(solver, options)

    
    # if options.logging, time the linear system solve
    if options.logging
        time_start = time_ns()
    end
    
    # update linear operator, i.e., ∇²f(xᵏ)
    update!(solver.lhs_op.Hf_xk, solver)
    linsys_tol = max(
        sqrt(eps()), min(
            sqrt(solver.rp_norm * solver.rd_norm), options.linsys_max_tol)
    )

    # warm start if past first iteration
    !isinf(solver.rp_norm) && warm_start!(linsys_solver, solver.xk)
    cg!(
        linsys_solver, solver.lhs_op, solver.cache.rhs;
        M=solver.P, rtol=linsys_tol
    )
    !issolved(linsys_solver) && error("CG failed")
    
    update_δx!(linsys_solver.x, solver, options)
    solver.xk .= linsys_solver.x

    if options.logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end

function update_Mx!(solver::MLSolver{T}, options::SolverOptions) where {T}
    if options.relax && sqrt(solver.rp_norm * solver.rd_norm) < options.relax_tol
        # TODO: using solver.zk vs solver.Mxk??
        @. solver.Mxk = options.α * solver.xk + (one(T) - options.α) * solver.zk
    else
        @. solver.Mxk = solver.xk
    end
    return nothing
end

function update_Mx!(solver::Solver, options::SolverOptions)
    T = eltype(solver.xk)
    if options.relax && sqrt(solver.rp_norm * solver.rd_norm) < options.relax_tol
        mul!(solver.cache.vm, solver.data.M, solver.xk)
        @. solver.Mxk = options.α * solver.cache.vm + (one(T) - options.α) * (solver.zk + solver.data.c)
    else
        mul!(solver.Mxk, solver.data.M, solver.xk)
    end
    return nothing
end

function update_z!(solver::Solver, options::SolverOptions)
    options.logging && (time_start = time_ns();)
    @. solver.cache.vm = solver.Mxk + solver.uk - solver.data.c
    
    # prox_{g/ρ}(v) = prox_{g/ρ}( Axᵏ⁺¹ - c + uᵏ)
    solver.data.prox_g!(solver.zk, solver.cache.vm, solver.ρ)
    return options.logging ? (time_ns() - time_start) / 1e9 : nothing
end

function update_z!(solver::MLSolver, options::SolverOptions)
    options.logging && (time_start = time_ns();)
    @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)
    
    @. solver.cache.vm = solver.Mxk + solver.uk - solver.data.c
    
    solver.zk .= soft_threshold.(solver.cache.vm, solver.λ1/solver.ρ)
    return options.logging ? (time_ns() - time_start) / 1e9 : nothing
end

function update_z!(solver::ConicSolver, options::SolverOptions)
    options.logging && (time_start = time_ns();)
    @. solver.cache.vm = solver.Mxk + solver.uk - solver.data.c

    project!(solver.zk, solver.data.K, solver.cache.vm)
    return options.logging ? (time_ns() - time_start) / 1e9 : nothing
end

function update_u!(solver::Solver, options::SolverOptions)
    @. solver.uk += solver.Mxk - solver.zk - solver.data.c
end

function update_u!(solver::ConicSolver, options::SolverOptions)
    @. solver.δy .= solver.ρ * ( solver.Mxk - solver.zk - solver.data.c )
    @. solver.uk += solver.δy / solver.ρ
end

function compute_primal_residual!(solver::Solver, options::SolverOptions)
    @. solver.rp = solver.Mxk - solver.zk - solver.data.c
    solver.rp_norm = norm(solver.rp, options.norm_type)
    return nothing
end

function compute_dual_residual!(solver::Solver, options::SolverOptions)
    solver.data.grad_f!(solver.cache.vn, solver.xk)
    mul!(solver.cache.vn2, solver.data.M', solver.uk)

    @. solver.rd = solver.cache.vn + solver.ρ * solver.cache.vn2
    solver.rd_norm = norm(solver.rd, options.norm_type)

    # dual residual option 2
    # @. solver.cache.vm = solver.zk_old - solver.zk
    # mul!(solver.rd, solver.data.M', solver.cache.vm)
    # @. solver.rd *= solver.ρ

    return nothing
end

function compute_dual_residual!(solver::ConicSolver, options::SolverOptions)
    mul!(solver.cache.vn, solver.data.P, solver.xk)
    solver.cache.vn .+= solver.data.q
    mul!(solver.cache.vn2, solver.data.M', solver.uk)

    @. solver.rd = solver.cache.vn + solver.ρ * solver.cache.vn2
    solver.rd_norm = norm(solver.rd, options.norm_type)
    return nothing
end

function compute_dual_residual!(solver::MLSolver, options::SolverOptions)
    @. solver.cache.vN = solver.data.df(solver.pred)
    mul!(solver.cache.vn, solver.data.Adata', solver.cache.vN)
    @. solver.cache.vn += solver.λ2 * solver.xk

    mul!(solver.cache.vn2, solver.data.M', solver.uk)

    @. solver.rd = solver.cache.vn + solver.ρ * solver.cache.vn2
    solver.rd_norm = norm(solver.rd, options.norm_type)
    return nothing
end

function compute_residuals!(solver::Solver, options::SolverOptions)
    compute_primal_residual!(solver, options)
    compute_dual_residual!(solver, options)
    return nothing
end

function infeasible(::Solver, ::SolverOptions)
    return false
end

# TODO: Update for general conic programs
function infeasible(solver::ConicSolver, options::SolverOptions)
    return primal_infeasible(solver, options) || dual_infeasible(solver, options)
end

# ------------------------------------------------------------------------------
# TODO: change to use scaled dual variable
function primal_infeasible(solver::ConicSolver, options::SolverOptions)
    isinf(solver.rp_norm) && return false
    vn = solver.cache.vn
    δy = solver.δy

    # TODO: figure out if need to add this
    all(iszero, δy) && return false

    δy_norm = norm(δy, Inf)
    mul!(vn, solver.data.M', δy)
    ATδy_norm = norm(vn, Inf)
    ATδy_norm > options.eps_inf * δy_norm && return false

    support(δy, solver.data.K) > options.eps_inf * δy_norm && return false

    return true    
end

function dual_infeasible(solver::ConicSolver, options::SolverOptions)
    isinf(solver.rp_norm) && return false
    vn, vm = solver.cache.vn, solver.cache.vm
    δx = solver.δx

    δx_norm = norm(δx, Inf)
    mul!(vn, solver.data.P, δx)
    Pδx_inf = norm(vn, Inf)
    Pδx_inf > options.eps_inf * δx_norm && return false

    dot(solver.data.q, δx) > options.eps_inf * δx_norm && return false

    mul!(vm, solver.data.M, δx)
    !in_recession_cone(vm, solver.data.K, options.eps_inf * δx_norm) && return false
    
    return true

end
# ------------------------------------------------------------------------------

function solve!(
    solver::Solver;
    options::SolverOptions=SolverOptions(),
)
    setup_time_start = time_ns()
    options.verbose && @printf("Starting setup...")

    # --- parameters & data ---
    m, n = solver.data.m, solver.data.n

    # --- setup ---
    t = 0
    solver.dual_gap = Inf
    solver.obj_val = Inf
    solver.loss = Inf
    solver.rp_norm = Inf
    solver.rd_norm = Inf
    solver.xk .= zeros(n)
    solver.Mxk .= zeros(m)
    solver.zk .= zeros(m)
    solver.uk .= zeros(m)
    # Computed variables
    obj_val!(solver, options)
    convergence_criteria!(solver, options)


    # --- enable multithreaded BLAS ---
    BLAS.set_num_threads(options.num_threads)

    # --- Setup Linear System Solver ---
    solver.ρ = options.ρ0
    solver.lhs_op.ρ[1] = options.ρ0
    precond_time = build_preconditioner!(solver, options)
    linsys_solver = CgSolver(n, n, typeof(solver.xk))

    # --- Logging ---
    if options.logging
        tmp_log = create_temp_log(solver, options.max_iters)
    end


    setup_time = (time_ns() - setup_time_start) / 1e9
    options.verbose && @printf("\nSetup in %6.3fs\n", setup_time)

    # --- Print Headers ---
    format = print_format(solver, options)
    headers = print_headers(solver, options)
    options.verbose && print_header(format, headers)

    # --- Print 0ᵗʰ iteration ---
    iter_fmt = iter_format(solver, options)
    options.verbose && print_iter_func(iter_fmt, iter_data(solver, options, 0, 0.0))

    # --------------------------------------------------------------------------
    # --------------------- ITERATIONS -----------------------------------------
    # --------------------------------------------------------------------------
    solve_time_start = time_ns()
    while t < options.max_iters && 
        (time_ns() - solve_time_start) / 1e9 < options.max_time_sec
        
        t += 1

        # --- ADMM iterations ---
        time_linsys = update_x!(solver, options, linsys_solver)
        update_Mx!(solver, options)
        # solver.zk_old .= solver.zk
        time_prox = update_z!(solver, options)
        update_u!(solver, options)
        
        # --- Update objective & convergence criteria ---
        # NOTE: obj_val! also updates pred = Adata*xk - bdata for MLSolver 
        obj_val!(solver, options)
        compute_residuals!(solver, options)
        convergence_criteria!(solver, options)

        # --- Logging ---
        time_sec = (time_ns() - solve_time_start) / 1e9
        if options.logging
            populate_log!(tmp_log, solver, options, t, time_sec, time_linsys, time_prox)
        end

        # --- Printing ---
        if options.verbose && (t == 1 || t % options.print_iter == 0)
            print_iter_func(
                iter_fmt,
                iter_data(solver, options, t, time_sec)
            )
        end

        # --- Check convergence & infeasibility ---
        converged(solver, options) && break
        t % options.infeas_check_iter == 0 && infeasible(solver, options) && break

        # --- Update ρ ---
        if t % options.rho_update_iter == 0
            updated_rho = update_rho!(solver, options)

            if options.precondition && updated_rho
                update_preconditioner_rho!(solver, options)
            end
        end

        # --- Update preconditioner ---
        if options.precondition && t % options.sketch_update_iter == 0
            update_preconditioner!(solver, options)
        end

    end

    # --- Print Final Iteration ---
    if options.verbose && (t % options.print_iter != 0 && t != 1)
        print_iter_func(
            iter_fmt,
            iter_data(solver, options, t, (time_ns() - solve_time_start) / 1e9)
        )
    end

    # --- Print Footer ---
    solve_time = (time_ns() - solve_time_start) / 1e9
    if !converged(solver, options)
        options.verbose && @printf("\nWARNING: did not converge after %d iterations, %6.3fs:", t, solve_time)
        if t >= options.max_iters
            options.verbose && @printf(" (max iterations reached)\n")
            status = :ITERATION_LIMIT
        elseif (time_ns() - solve_time_start) / 1e9 >= options.max_time_sec
            options.verbose && @printf(" (max time reached)\n")
            status = :TIME_LIMIT
        elseif primal_infeasible(solver, options)
            options.verbose && @printf(" (primal infeasible problem detected)\n")
            status = :INFEASIBLE
        elseif dual_infeasible(solver, options)
            options.verbose && @printf(" (dual infeasible problem detected)\n")
            status = :DUAL_INFEASIBLE
        end
    else
        options.verbose && @printf("\nSOLVED in %6.3fs, %d iterations\n", solve_time, t)
        options.verbose && @printf("Total time: %6.3fs\n", setup_time + solve_time)
        status = :OPTIMAL
    end 
    options.verbose && print_footer()


    # --- Construct Logs ---
    if options.logging
        log = GeNIOSLog(
            tmp_log.dual_gap[1:t-1],
            tmp_log.obj_val[1:t-1],
            tmp_log.iter_time[1:t-1],
            tmp_log.linsys_time[1:t-1],
            tmp_log.prox_time[1:t-1],
            tmp_log.rp[1:t-1], 
            tmp_log.rd[1:t-1],
            setup_time, 
            precond_time, 
            solve_time
        )
    else
        log = GeNIOSLog(setup_time, precond_time, solve_time)
    end

    # --- Construct Solution ---
    res = GeNIOSResult(
        status,
        solver.obj_val,
        solver.loss,
        solver.xk,
        solver.zk,          # usually want zk since it is feasible (or sparsified)
        solver.uk,
        solver.dual_gap,
        log
    )

    return res
end