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

function print_format(::Solver)
    return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s"]
end

function print_headers(::Solver)
    return ["Iteration", "Obj Val", "r_primal", "r_dual", "ρ", "Time"]
end

function iter_format(::Solver)
    return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
end

function iter_data(solver::Solver, t, time_sec)
    return (
        string(t),
        solver.obj_val,
        solver.rp_norm,
        solver.rd_norm,
        solver.ρ,
        time_sec
    )
end

function print_format(::MLSolver)
    return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s"]
end

function print_headers(::MLSolver)
    return ["Iteration", "Objective", "RMSE", "Dual Gap", "r_primal", "r_dual", "ρ", "Time"]
end

function iter_format(::MLSolver)
    return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
end

function iter_data(solver::MLSolver, t, time_sec)
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
end