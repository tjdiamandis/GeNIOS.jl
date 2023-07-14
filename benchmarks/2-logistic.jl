using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2
using CSV, DataFrames, Statistics, JLD2
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE_DENSE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const SAVEPATH = joinpath(@__DIR__, "saved")
const FIGS_PATH = joinpath(@__DIR__, "figures")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA_SPARSE = true
const RAN_TRIALS = false


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# m, n = 10_000, 20_000
# A,b = get_augmented_data(m, n, DATAFILE_DENSE)
# b[b .< mean(b)] .= -1
# b[b .>= mean(b)] .= 1

# # Reguarlization parameters
# λ1_max = norm(0.5*A'*ones(m), Inf)
# λ1 = 0.05*λ1_max
# λ2 = 0.0

# ## Solving the Problem
# # For compilation
# solve!(GeNIOS.LogisticSolver(λ1, λ2, A, b); options=GeNIOS.SolverOptions(max_iters=2, use_dual_gap=true))

# # With everything
# solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
# options = GeNIOS.SolverOptions(
#     use_dual_gap=true,
#     dual_gap_tol=1e-2,
#     precondition=true,
#     num_threads=Sys.CPU_THREADS,
#     ρ0=10.0,
#     rho_update_iter=4000,
#     max_iters=5000,
# )
# GC.gc()
# result = solve!(solver; options=options)


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

function run_trial_genios(; type, m=10_000, n=20_000)
    BLAS.set_num_threads(Sys.CPU_THREADS)
    if type == "sparse"
        A, b = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE)
    elseif type == "dense"
        A, b = get_augmented_data(m, n, DATAFILE_DENSE)
        b[b .< mean(b)] .= -1
        b[b .>= mean(b)] .= 1
    else
        error("Unknown type: $type")
    end

    # Reguarlization parameters
    λ1_max = norm(A'*b, Inf)
    λ1 = 0.1*λ1_max
    λ2 = 0.0

    ## Solving the Problem
    # For compilation
    solve!(GeNIOS.LogisticSolver(λ1, λ2, A, b); options=GeNIOS.SolverOptions(max_iters=2, use_dual_gap=true))

    # With everything
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-1,
        precondition=true,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=60*30.
    )
    GC.gc()
    result = solve!(solver; options=options)

    # No preconditioner
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-1,
        verbose=true,
        precondition=false,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=60*30.
    )
    GC.gc()
    result_npc = solve!(solver; options=options)

    # Exact solve
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-1,
        verbose=true,
        precondition=true,
        linsys_max_tol=1e-8,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=60*30.
    )
    GC.gc()
    result_exact = solve!(solver; options=options)

    # Exact solve, no pc
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-1,
        verbose=true,
        precondition=false,
        linsys_max_tol=1e-8,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=60*30.
    )
    GC.gc()
    result_exact_npc = solve!(solver; options=options)

    # High precision solve
    if type == "sparse"
        solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
        options = GeNIOS.SolverOptions(
            # relax=true,
            # α=1.6,
            use_dual_gap=true,
            # dual_gap_tol = sqrt(eps()),
            dual_gap_tol = 1e-11,
            verbose=true,
            precondition=true,
            num_threads=Sys.CPU_THREADS,
            ρ0=10.0,
            rho_update_iter=1000,
            max_iters=5000,
        )
        GC.gc()
        result_high_precision = solve!(solver; options=options)
    else
        result_high_precision = nothing
    end


    savefile = joinpath(SAVEPATH, "2-logistic-$type.jld2")
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
    @info "Starting trials..."
    # for type in ["sparse", "dense"]
    for type in ["dense"]
        run_trial(type=type)
        @info "Finished with type=$type"
    end
    @info "Finished with all trials"
end

#
for type in ["sparse", "dense"]
# for type in ["dense"]
    savefile = joinpath(SAVEPATH, "2-logistic-$type.jld2")
    result, result_npc, result_exact, result_exact_npc, result_high_precision = 
        load(savefile, 
            "result", "result_npc", "result_exact", "result_exact_npc", "result_high_precision"
        )
    any(!isnothing(x) && x.status != :OPTIMAL for x in (
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
    if type == "sparse"
        log_high_precision = result_high_precision.log
        pstar = log_high_precision.obj_val[end]
    end

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
        legend=:topright,
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
        legend=:topright,
    )
    add_to_plot!(rd_iter_plot, log.iter_time, log.rd, "GeNIOS", :coral);
    add_to_plot!(rd_iter_plot, log_npc.iter_time, log_npc.rd, "No PC", :purple);
    add_to_plot!(rd_iter_plot, log_exact.iter_time, log_exact.rd, "ADMM (pc)", :red);
    add_to_plot!(rd_iter_plot, log_exact_npc.iter_time, log_exact_npc.rd, "ADMM (no pc)", :mediumblue);
    rd_iter_plot

    if type == "sparse"
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
            legend=:topright,
        )
        add_to_plot!(obj_val_iter_plot, log.iter_time, (log.obj_val .- pstar) ./ pstar, "GeNIOS", :coral);
        add_to_plot!(obj_val_iter_plot, log_npc.iter_time, (log_npc.obj_val .- pstar) ./ pstar, "No PC", :purple);
        add_to_plot!(obj_val_iter_plot, log_exact.iter_time, (log_exact.obj_val .- pstar) ./ pstar, "ADMM (pc)", :red);
        add_to_plot!(obj_val_iter_plot, log_exact_npc.iter_time, (log_exact_npc.obj_val .- pstar) ./ pstar, "ADMM (no pc)", :mediumblue);
        obj_val_iter_plot
        savefig(obj_val_iter_plot, joinpath(FIGS_PATH, "2-logistic-obj-val-iter-$type.pdf"))

        
        high_precision_plt = plot(; 
        dpi=300,
        # title="Convergence (Lasso)",
        yaxis=:log,
        xlabel="Iteration",
        legend=:topright,
        ylims=(1e-10, 1000),
        legendfontsize=14,
        labelfontsize=14,
        titlefontsize=14
        )
        add_to_plot!(high_precision_plt, log_high_precision.iter_time, log_high_precision.rp, "Primal Residual", :indigo)
        add_to_plot!(high_precision_plt, log_high_precision.iter_time, log_high_precision.rd, "Dual Residual", :red)
        add_to_plot!(high_precision_plt, log_high_precision.iter_time, log_high_precision.dual_gap, "Duality Gap", :mediumblue)
        add_to_plot!(high_precision_plt, log_high_precision.iter_time, sqrt(eps())*ones(length(log_high_precision.iter_time)), L"\sqrt{\texttt{eps}}", :black; style=:dash, lw=1)
        savefig(high_precision_plt, joinpath(FIGS_PATH, "2-logistic-high-precision-$type.pdf"))
    end
    
    savefig(dual_gap_iter_plt, joinpath(FIGS_PATH, "2-logistic-dual-gap-iter-$type.pdf"))
    savefig(rp_iter_plot, joinpath(FIGS_PATH, "2-logistic-rp-iter-$type.pdf"))
    savefig(rd_iter_plot, joinpath(FIGS_PATH, "2-logistic-rd-iter-$type.pdf"))
end