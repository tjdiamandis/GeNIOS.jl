# --- Solver printing utils ---
function print_header(format, data)
    @printf(
        "\n──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
end


function print_footer()
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
    )
end


function print_iter_func(format, data)
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
end

function print_format(::Solver, ::SolverOptions)
    return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s"]
end

function print_headers(::Solver, ::SolverOptions)
    return ["Iteration", "Obj Val", "r_primal", "r_dual", "ρ", "Time"]
end

function iter_format(::Solver, ::SolverOptions)
    return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
end

function iter_data(solver::Solver, ::SolverOptions, t, time_sec)
    return (
        string(t),
        solver.obj_val,
        solver.rp_norm,
        solver.rd_norm,
        solver.ρ,
        time_sec
    )
end

function print_format(::MLSolver, options::SolverOptions)
    if options.use_dual_gap
        return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s"]
    else
        return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s"]
    end
end

function print_headers(::MLSolver, options::SolverOptions)
    if options.use_dual_gap
        return ["Iteration", "Objective", "RMSE", "Dual Gap", "r_primal", "r_dual", "ρ", "Time"]
    else
        return ["Iteration", "Objective", "RMSE", "r_primal", "r_dual", "ρ", "Time"]
    end
end

function iter_format(::MLSolver, options::SolverOptions)
    if options.use_dual_gap
        return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
    else
        return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
    end
end

function iter_data(solver::MLSolver, options::SolverOptions, t, time_sec)
    if options.use_dual_gap
        return (
            string(t),
            solver.obj_val,
            sqrt(solver.loss / solver.data.N),
            solver.dual_gap,
            solver.rp_norm,
            solver.rd_norm,
            solver.ρ,
            time_sec
        )
    else
        return (
            string(t),
            solver.obj_val,
            sqrt(solver.loss / solver.data.N),
            solver.rp_norm,
            solver.rd_norm,
            solver.ρ,
            time_sec
        )
    end
end
