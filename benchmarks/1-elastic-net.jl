using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2
using CSV, DataFrames, Statistics, JLD2
include(joinpath(@__DIR__, "utils.jl"))
using GeNIOS

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE_DENSE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const SAVEPATH = joinpath(@__DIR__, "saved")
const FIGS_PATH = joinpath(@__DIR__, "figures")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA_SPARSE = true
const RAN_TRIALS = true


function run_trial(; type, m=10_000, n=20_000)
    if type == "sparse"
        A, b = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE)
    elseif type == "dense"
        A, b = get_augmented_data(m, n, DATAFILE_DENSE)
    else
        error("Unknown type: $type")
    end

    # Reguarlization parameters
    λ1_max = norm(A'*b, Inf)
    λ1 = 0.1*λ1_max
    λ2 = λ1

    # For compilation
    solve!(GeNIOS.ElasticNetSolver(λ1, λ2, A, b); options=GeNIOS.SolverOptions(use_dual_gap=true, max_iters=2, verbose=false))

    # With everything
    solver = GeNIOS.ElasticNetSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        verbose=false,
        precondition=true,
        sketch_update_iter=1000,    # We know that the Hessian AᵀA does not change
        ρ0=10.0,
        rho_update_iter=1000,
    )
    GC.gc()
    result = solve!(solver; options=options)

    # No preconditioner
    solver = GeNIOS.ElasticNetSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        verbose=false,
        precondition=false,
        sketch_update_iter=1000,    # We know that the Hessian AᵀA does not change
        ρ0=10.0,
        rho_update_iter=1000,
    )
    GC.gc()
    result_npc = solve!(solver; options=options)

    # Exact solve
    solver = GeNIOS.ElasticNetSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        verbose=false,
        precondition=true,
        linsys_max_tol=1e-8,
        sketch_update_iter=1000,    # We know that the Hessian AᵀA does not change
        ρ0=10.0,
        rho_update_iter=1000,
    )
    GC.gc()
    result_exact = solve!(solver; options=options)

    # Exact solve, no pc
    solver = GeNIOS.ElasticNetSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        verbose=false,
        precondition=false,
        linsys_max_tol=1e-8,
        sketch_update_iter=1000,    # We know that the Hessian AᵀA does not change
        ρ0=10.0,
        rho_update_iter=1000,
    )
    GC.gc()
    result_exact_npc = solve!(solver; options=options)

    # High precision solve
    solver = GeNIOS.ElasticNetSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        use_dual_gap=true,
        α=1.6,
        eps_abs = 1e-10,
        eps_rel = 1e-10,
        dual_gap_tol = 10eps(),
        verbose=false,
        precondition=true,
        sketch_update_iter=1000,    # We know that the Hessian AᵀA does not change
        ρ0=10.0,
        rho_update_iter=1000,
    )
    GC.gc()
    result_high_precision = solve!(solver; options=options)

    savefile = joinpath(SAVEPATH, "1-elastic-net-$type.jld2")
    save(savefile, 
        "result", result,
        "result_npc", result_npc,
        "result_exact", result_exact,
        "result_exact_npc", result_exact_npc,
        "result_high_precision", result_high_precision
    )

    return nothing
end


if !RAN_TRIALS
    for type in ["sparse", "dense"]
        run_trial(type=type)
        @info "Finished with type=$type"
    end
end
@info "Finished with all trials"


for type in ["sparse", "dense"]
    savefile = joinpath(SAVEPATH, "1-elastic-net-$type.jld2")
    result, result_npc, result_exact, result_exact_npc, result_high_precision = 
        load(savefile, 
            "result", "result_npc", "result_exact", "result_exact_npc", "result_high_precision"
        )
    any(x.status != :OPTIMAL for x in (
        result,
        result_npc,
        result_exact,
        result_exact_npc,
        result_high_precision
    )) && @warn "Some problems not solved! for $type"

    log = result.log
    log_npc = result_npc.log
    log_exact = result_exact.log
    log_exact_npc = result_exact_npc.log
    log_high_precision = result_high_precision.log
    pstar = log_high_precision.obj_val[end]

    names = ["GeNIOS", "GeNIOS (no pc)", "ADMM", "ADMM (no pc)"]
    logs = [log, log_npc, log_exact, log_exact_npc]
    @info "\n\n ----- TIMING TABLE FOR $type -----\n"
    print_timing_table(names, logs)

    # Plot things
    dual_gap_iter_plt = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel="Dual Gap",
    xlabel="Time (s)",
    legend=:topright,
    )
    add_to_plot!(dual_gap_iter_plt, log.iter_time, log.dual_gap, "GeNIOS", :coral);
    add_to_plot!(dual_gap_iter_plt, log_npc.iter_time, log_npc.dual_gap, "No PC", :purple);
    add_to_plot!(dual_gap_iter_plt, log_exact.iter_time, log_exact.dual_gap, "ADMM (pc)", :red);
    add_to_plot!(dual_gap_iter_plt, log_exact_npc.iter_time, log_exact_npc.dual_gap, "ADMM (no pc)", :mediumblue);
    dual_gap_iter_plt

    rp_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Primal Residual $\ell_2$ Norm",
    xlabel="Time (s)",
    legend=false,
    )
    add_to_plot!(rp_iter_plot, log.iter_time, log.rp, "GeNIOS", :coral);
    add_to_plot!(rp_iter_plot, log_npc.iter_time, log_npc.rp, "No PC", :purple);
    add_to_plot!(rp_iter_plot, log_exact.iter_time, log_exact.rp, "ADMM (pc)", :red);
    add_to_plot!(rp_iter_plot, log_exact_npc.iter_time, log_exact_npc.rp, "ADMM (no pc)", :mediumblue);
    rp_iter_plot

    rd_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Dual Residual $\ell_2$ Norm",
    xlabel="Time (s)",
    legend=false,
    )
    add_to_plot!(rd_iter_plot, 1:length(log.iter_time), log.rd, "GeNIOS", :coral);
    add_to_plot!(rd_iter_plot, 1:length(log_npc.iter_time), log_npc.rd, "No PC", :purple);
    add_to_plot!(rd_iter_plot, 1:length(log_exact.iter_time), log_exact.rd, "ADMM (pc)", :red);
    add_to_plot!(rd_iter_plot, 1:length(log_exact_npc.iter_time), log_exact_npc.rd, "ADMM (no pc)", :mediumblue);
    rd_iter_plot

    pstar = minimum(vcat(
    log.obj_val,
    log_npc.obj_val,
    log_exact.obj_val,
    log_exact_npc.obj_val,
    log_high_precision.obj_val
    )) - sqrt(eps())
    obj_val_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"$(p-p^\star)/p^\star$",
    xlabel="Time (s)",
    legend=false,
    )
    add_to_plot!(obj_val_iter_plot, log.iter_time, (log.obj_val .- pstar) ./ pstar, "GeNIOS", :coral);
    add_to_plot!(obj_val_iter_plot, log_npc.iter_time, (log_npc.obj_val .- pstar) ./ pstar, "No PC", :purple);
    add_to_plot!(obj_val_iter_plot, log_exact.iter_time, (log_exact.obj_val .- pstar) ./ pstar, "ADMM (pc)", :red);
    add_to_plot!(obj_val_iter_plot, log_exact_npc.iter_time, (log_exact_npc.obj_val .- pstar) ./ pstar, "ADMM (no pc)", :mediumblue);
    obj_val_iter_plot

    savefig(dual_gap_iter_plt, joinpath(FIGS_PATH, "1-elastic-net-dual-gap-iter-$type.pdf"))
    savefig(rp_iter_plot, joinpath(FIGS_PATH, "1-elastic-net-rp-iter-$type.pdf"))
    savefig(rd_iter_plot, joinpath(FIGS_PATH, "1-elastic-net-rd-iter-$type.pdf"))
    savefig(obj_val_iter_plot, joinpath(FIGS_PATH, "1-elastic-net-obj-val-iter-$type.pdf"))
    

    high_precision_plt = plot(; 
        dpi=300,
        # title="Convergence (Lasso)",
        yaxis=:log,
        xlabel="Iteration",
        legend=(type == "sparse" ? :topright : false),
        ylims=(1e-10, 1000),
        legendfontsize=18,
        labelfontsize=18,
        titlefontsize=18,
        tickfontsize=12
    )
    add_to_plot!(high_precision_plt, log_high_precision.iter_time, log_high_precision.rp, "Primal Residual", :indigo)
    add_to_plot!(high_precision_plt, log_high_precision.iter_time, log_high_precision.rd, "Dual Residual", :red)
    add_to_plot!(high_precision_plt, log_high_precision.iter_time, log_high_precision.dual_gap, "Duality Gap", :mediumblue)
    add_to_plot!(high_precision_plt, log_high_precision.iter_time, sqrt(eps())*ones(length(log_high_precision.iter_time)), L"\sqrt{\texttt{eps}}", :black; style=:dash, lw=1)
    savefig(high_precision_plt, joinpath(FIGS_PATH, "1-elastic-net-high-precision-$type.pdf"))
end
