function build_preconditioner!(solver::Solver, options::SolverOptions)
    precond_time_start = time_ns()
    return (time_ns() - precond_time_start) / 1e9
end

function compute_residuals(solver::Solver, options::SolverOptions)
    A = solver.A
    x, z, z_old, = solver.xk, solver.zk, solver.zk_old
    rp, rd = solver.rp, solver.rd
    rp_norm, rd_norm = solver.rp_norm, solver.rd_norm
    
    # primal residual
    mul!(rp, A, x)
    @. rp -= z
    rp_norm = norm(rp, options.norm_type)

    # dual residual
    @. solver.cache.vm = z - z_old
    mul!(rd, A', solver.cache.m)
    rd_norm = norm(rd, options.norm_type)

    return nothing
end

function update_rho!(solver::Solver)
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

    return nothing
end

# TODO: extend for MLSolver
function converged(solver::Solver, options::SolverOptions)
    mul!(solver.cache.vm, solver.data.A, solver.xk)
    norm_Ax = norm(solver.cache.vm)
    norm_z = norm(solver.zk)
    norm_c = norm(solver.data.c)
    eps_pri = sqrt(solver.data.m) * options.eps_abs + options.eps_rel * max(norm_Ax, norm_z, norm_c)

    mul!(solver.cache.vn, solver.data.A', solver.uk)
    norm_Aty = norm(solver.cache.vn)
    eps_dual = sqrt(solver.data.n) * options.eps_abs + options.eps_rel * norm_Aty

    return solver.rp_norm ≤ eps_pri && solver.rd_norm ≤ eps_dual
end

# Used to implement custom convergence criteria (e.g., via dual gap)
function convergence_criteria!(solver::Solver, options::SolverOptions)
    return nothing
end

# Comptues dual gap TODO:
function convergence_criteria!(solver::MLSolver, options::SolverOptions)
    return nothing
end

function obj_val!(solver::Solver)
    solver.obj_val = solver.data.f(solver.xk) + solver.data.g(solver.zk)
end

function obj_val!(solver::MLSolver)
    # pred = Ax - b
    mul!(solver.pred, solver.data.Adata, solver.zk)
    solver.pred .-= solver.data.bdata
    
    solver.obj_val = sum(x->solver.data.f(x), solver.pred) + 
        solver.λ1*norm(solver.zk, 1) + solver.λ1*sum(abs2, solver.zk)
end

function compute_rhs!(solver::Solver)
    # RHS = ∇²f(xᵏ)xᵏ - ∇f(xᵏ) - ρAᵀ(zᵏ - c + uᵏ)
    mul!(solver.cache.vn, solver.data.Hf, solver.xk)
    solver.data.grad_f!(solver.cache.vn2, solver.xk)
    
    @. solver.cache.vm = solver.zk - solver.data.c + solver.uk
    mul!(solver.cache.rhs, solver.data.A', solver.cache.vm)
    @. solver.cache.rhs = solver.cache.vn - solver.cache.vn2 - solver.ρ * solver.cache.rhs
    
    return nothing
end

# NOTE: Assumes that pred has been computed
function compute_rhs!(solver::MLSolver)
    # RHS = ∇²f(xᵏ)xᵏ - ∇f(xᵏ) - ρAᵀ(zᵏ - c + uᵏ)

    # compute first term (hessian)
    mul!(solver.cache.vN, solver.data.Adata, solver.xk)
    @. solver.cache.vN *= solver.data.d2f(solver.pred)
    mul!(solver.cache.vn, solver.data.Adata', solver.cache.vN)

    # compute second term (gradient)
    @. solver.cache.vN = solver.data.df(solver.pred)
    mul!(solver.cache.vn2, solver.data.Adata', solver.cache.vN)
    
    # compute last term
    @. solver.cache.vm = solver.zk - solver.data.c + solver.uk
    mul!(solver.cache.rhs, solver.data.A', solver.cache.vm)

    # add them up
    @. solver.cache.rhs = solver.cache.vn - solver.cache.vn2 - solver.ρ * solver.cache.rhs
    
    return nothing
end


function update_x!(
    solver::Solver,
    options::SolverOptions,
    linsys_solver::CgSolver
)
    # RHS = ∇²f(xᵏ)xᵏ - ∇f(xᵏ) - ρAᵀ(zᵏ - c + uᵏ)
    compute_rhs!(solver)

    
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
    solver.xk .= linsys_solver.x

    if options.logging 
        return (time_ns() - time_start) / 1e9
    else
        return nothing
    end
end

function update_Ax!(solver::MLSolver{T}, options::SolverOptions) where {T}
    if options.relax
        @. solver.cache.vm = -solver.xk
        @. solver.Axk = solver.α * solver.cache.vm + (one(T) - solver.α) * solver.zk
    else
        @. solver.Axk = -solver.xk
    end
    return nothing
end

function update_Ax!(solver::GenericSolver{T}, options::SolverOptions) where {T}
    if options.relax
        mul!(solver.cache.vm, solver.data.A, solver.xk)
        @. solver.Axk = solver.α * solver.cache.vm + (one(T) - solver.α) * solver.zk
    else
        mul!(solver.Axk, solver.data.A, solver.xk)
    end
    return nothing
end

function update_z!(solver::Solver, options::SolverOptions)
    solver.cache.vm .= -solver.Axk
    @. solver.cache.vm += -solver.uk + solver.data.c
    
    # prox_{g/ρ}(v) = prox_{g/ρ}( -(Axᵏ⁺¹ - c + uᵏ⁺¹) )
    solver.data.prox_g!(solver.zk, solver.cache.vm, solver.ρ)
end

function update_z!(solver::MLSolver, options::SolverOptions)
    @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)
    
    solver.cache.vm .= -solver.Axk
    @. solver.cache.vm += -solver.uk + solver.data.c
    
    solver.zk .= soft_threshold.(solver.cache.vm, solver.λ1/solver.ρ)
end

function update_u!(solver::Solver, options::SolverOptions)
    @. solver.uk = solver.uk + solver.Axk + solver.zk - solver.data.c
end

function compute_residuals!(solver::Solver, options::SolverOptions)
    # primal residual
    @. solver.rp = solver.Axk + solver.zk - solver.data.c
    solver.rp_norm = norm(solver.rp, options.norm_type)

    # dual residual
    @. solver.cache.vm = solver.zk - solver.zk_old
    mul!(solver.rd, solver.data.A', solver.cache.vm)
    @. solver.rd *= solver.ρ
    solver.rd_norm = norm(solver.rd, options.norm_type)
end

function solve!(
    solver::Solver;
    options::SolverOptions=SolverOptions(),
)
    setup_time_start = time_ns()
    options.verbose && @printf("Starting setup...")

    # --- parameters & data ---
    m, n = solver.data.m, solver.data.n

    # --- setup ---
    t = 1
    solver.dual_gap = Inf
    solver.obj_val = Inf
    solver.loss = Inf
    solver.rp_norm = Inf
    solver.rd_norm = Inf
    solver.xk .= zeros(n)
    solver.Axk .= zeros(m)
    solver.zk .= zeros(m)
    solver.uk .= zeros(m)


    # --- enable multithreaded BLAS ---
    if options.multithreaded
        BLAS.set_num_threads(Sys.CPU_THREADS)
    else
        BLAS.set_num_threads(1)
    end

    # --- Setup Linear System Solver ---
    precond_time = build_preconditioner!(solver, options)
    linsys_solver = CgSolver(n, n, typeof(solver.xk))

    # --- Logging ---
    if options.logging
        tmp_log = create_temp_log(solver, options.max_iters)
    end


    setup_time = (time_ns() - setup_time_start) / 1e9
    options.verbose && @printf("\nSetup in %6.3fs\n", setup_time)

    # --- Print Headers ---
    # TODO: allow for custom headers
    format = ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s"]
    headers = ["Iteration", "Obj Val", "r_primal", "r_dual", "ρ", "Time"]
    options.verbose && print_header(format, headers)

    # --- Print 0ᵗʰ iteration ---
    obj_val!(solver)
    # TODO: dual_gap!(solver)
    iter_fmt = ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
    options.verbose && print_iter_func(iter_fmt, (
        string(0), solver.obj_val, Inf, Inf, solver.ρ, 0.0
    ))

    # --------------------------------------------------------------------------
    # --------------------- ITERATIONS -----------------------------------------
    # --------------------------------------------------------------------------
    solve_time_start = time_ns()
    while t <= options.max_iters && 
        (time_ns() - solve_time_start) / 1e9 < options.max_time_sec &&
        !converged(solver, options)


        # --- ADMM iterations ---
        time_linsys = update_x!(solver, options, linsys_solver)
        update_Ax!(solver, options)
        solver.zk_old .= solver.zk
        update_z!(solver, options)
        update_u!(solver, options)


        # --- Update ρ ---
        compute_residuals!(solver, options)
        if t % options.rho_update_iter == 0
            ρ_old = solver.ρ
            update_rho!(solver)
            
            # if updated_rho && !indirect && typeof(solver) <: LassoSolver
            #     # NOTE: logistic solver recomputes fact at each iteration anyway
            #     KKT_mat[diagind(KKT_mat)] .+= (solver.ρ .- ρ_old)
            #     linsys_solver = cholesky(KKT_mat)
            # elseif updated_rho && precondition && !(sketch_solve_x_update || gd_x_update)
            #     reg_term = typeof(solver) <: LogisticSolver ? solver.ρ : solver.ρ + solver.μ
            #     P = RP.NystromPreconditionerInverse(P.A_nys, reg_term)
            # end
        end



         # --- Update preconditioner ---
         if options.precondition && t % options.sketch_update_iter == 0
            # TODO:
            # update_preconditioner!(solver)
        end


        # --- Update objective & dual gap ---
        # Also updates pred = Adata*xk - bdata for MLSolver 
        obj_val!(solver)
        convergence_criteria!(solver, options)


        # --- Logging ---
        time_sec = (time_ns() - solve_time_start) / 1e9
        if options.logging
            # TODO: dual gap??
            # TODO: functionize
            # tmp_log.dual_gap[t] = solver.dual_gap
            tmp_log.obj_val[t] = solver.obj_val
            tmp_log.iter_time[t] = time_sec
            tmp_log.linsys_time[t] = time_linsys
            tmp_log.rp[t] = solver.rp_norm
            tmp_log.rd[t] = solver.rd_norm
        end

        # --- Printing ---
        if options.verbose && (t == 1 || t % options.print_iter == 0)
            # TODO: customization -- take a type of solver
            # TODO: functionize
            print_iter_func(
                iter_fmt,
                (
                string(t),
                solver.obj_val,
                solver.rp_norm,
                solver.rd_norm,
                solver.ρ,
                time_sec
                ))
        end

        t += 1
    end

    # --- Print Final Iteration ---
    if options.verbose && ((t-1) % options.print_iter != 0 && (t-1) != 1)
        print_iter_func(
            iter_fmt,
            (
            string(t),
            solver.obj_val,
            solver.rp_norm,
            solver.rd_norm,
            solver.ρ,
            (time_ns() - solve_time_start) / 1e9
        ))
    end

    # --- Print Footer ---
    solve_time = (time_ns() - solve_time_start) / 1e9
    options.verbose && @printf("\nSolved in %6.3fs, %d iterations\n", solve_time, t-1)
    options.verbose && @printf("Total time: %6.3fs\n", setup_time + solve_time)
    options.verbose && print_footer()


    # --- Construct Logs ---
    if options.logging
        log = NysADMMLog(
            tmp_log.dual_gap[1:t-1],
            tmp_log.obj_val[1:t-1],
            tmp_log.iter_time[1:t-1],
            tmp_log.linsys_time[1:t-1],
            tmp_log.rp[1:t-1], 
            tmp_log.rd[1:t-1],
            setup_time, 
            precond_time, 
            solve_time
        )
    else
        log = NysADMMLog(setup_time, precond_time, solve_time)
    end


    # --- Construct Solution ---
    res = NysADMMResult(
        solver.obj_val,
        solver.loss,
        solver.xk,
        solver.zk,          # usually want zk since it is feasible (or sparsified)
        solver.dual_gap,
        log
    )

    return res
end