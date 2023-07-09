using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using CSV, DataFrames, Statistics, JLD2
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

DATAPATH = joinpath(@__DIR__, "data")
DATAFILE = joinpath(DATAPATH, "YearPredictionMSD.txt")
SAVEPATH = joinpath(@__DIR__, "saved")
SAVEFILE = joinpath(SAVEPATH, "4-constrained-ls.jld2")
FIGS_PATH = joinpath(@__DIR__, "figures")

n = 10_000
m, 2n
P, q, A, l, u = construct_problem_constrained_ls(get_augmented_data(m, n, DATAFILE)...)

# compile
run_genios_trial_qp(P, q, A, l, u, options=GeNIOS.SolverOptions(max_iters=2))

# With everything
options = GeNIOS.SolverOptions(
    precondition=true,
    sketch_update_iter=10_000
    num_threads=1,
    eps_abs=1e-5,
    eps_rel=1e-5,
)
result = run_genios_trial_qp(P, q, A, l, u; options=options)

# No preconditioner
options_npc = GeNIOS.SolverOptions(
    precondition=false,
    num_threads=1,
    eps_abs=1e-5,
    eps_rel=1e-5,
)
result_npc = run_genios_trial_qp(P, q, A, l, u; options=options_npc)

# Exact solve
options_exact = GeNIOS.SolverOptions(
    precondition=true,
    sketch_update_iter=10_000
    num_threads=1,
    linsys_max_tol=1e-8,
    eps_abs=1e-5,
    eps_rel=1e-5,
)
result_exact = run_genios_trial_qp(P, q, A, l, u; options=options_exact)

# Exact solve, no pc
options_exact_npc = GeNIOS.SolverOptions(
    precondition=false,
    num_threads=1,
    linsys_max_tol=1e-8,
    eps_abs=1e-5,
    eps_rel=1e-5,
)
result_exact_npc = run_genios_trial_qp(P, q, A, l, u; options=options_exact_npc)

# High precision solve
options_high_precision = GeNIOS.SolverOptions(
    eps_abs=1e-8,
    eps_rel=1e-8,
    precondition=true,
    num_threads=Sys.CPU_THREADS,
)
result_high_precision = run_genios_trial_qp(P, q, A, l, u; options=options_high_precision)

save(SAVEFILE, 
    "result", result,
    "result_npc", result_npc,
    "result_exact", result_exact,
    "result_exact_npc", result_exact_npc,
    "result_high_precision", result_high_precision
)


## Load data from save files
result, result_npc, result_exact, result_exact_npc, result_high_precision = 
    load(SAVEFILE, 
        "result", "result_npc", "result_exact", "result_exact_npc", "result_high_precision"
    )

results = [
    result,
    result_npc,
    result_exact,
    result_exact_npc,
    result_high_precision
]
any(x.status != :OPTIMAL for x in results) && @warn "Some problems not solved!"

## Plots! Plots! Plots!
log_pc = result.log
log_npc = result_npc.log
log_exact = result_exact.log
log_exact_npc = result_exact_npc.log
log_high_precision = result_high_precision.log
pstar = log_high_precision.obj_val[end]

# Timing table
names = ["GeNIOS", "GeNIOS (no pc)", "ADMM", "ADMM (no pc)"]
logs = [log_pc, log_npc, log_exact, log_exact_npc]
print_timing_table(names, logs)

# Plot things
rp_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Primal Residual $\ell_2$ Norm",
    xlabel="Time (s)",
    legend=:topright,
)
add_to_plot!(rp_iter_plot, log_pc.iter_time, log_pc.rp, "GeNIOS", :coral);
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
add_to_plot!(rd_iter_plot, log_pc.iter_time, log_pc.rd, "GeNIOS", :coral);
add_to_plot!(rd_iter_plot, log_npc.iter_time, log_npc.rd, "No PC", :purple);
add_to_plot!(rd_iter_plot, log_exact.iter_time, log_exact.rd, "ADMM (pc)", :red);
add_to_plot!(rd_iter_plot, log_exact_npc.iter_time, log_exact_npc.rd, "ADMM (no pc)", :mediumblue);
rd_iter_plot

# NOTE: objective value is not with a feasible point; it is not very meaningful
pstar = log_high_precision.obj_val[end]
obj_val_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"$(p-p^\star)/p^\star$",
    xlabel="Time (s)",
    legend=:topright,
)
add_to_plot!(obj_val_iter_plot, log_pc.iter_time[2:end], abs.(log_pc.obj_val[2:end] .- pstar) ./ abs(pstar), "GeNIOS", :coral);
add_to_plot!(obj_val_iter_plot, log_npc.iter_time[2:end], abs.(log_npc.obj_val[2:end] .- pstar) ./ abs(pstar), "No PC", :purple);
add_to_plot!(obj_val_iter_plot, log_exact.iter_time[2:end], abs.(log_exact.obj_val[2:end] .- pstar) ./ abs(pstar), "ADMM (pc)", :red);
add_to_plot!(obj_val_iter_plot, log_exact_npc.iter_time[2:end], abs.(log_exact_npc.obj_val[2:end] .- pstar) ./ abs(pstar), "ADMM (no pc)", :mediumblue);
obj_val_iter_plot

savefig(rp_iter_plot, joinpath(FIGS_PATH, "4-constrained-ls-rp-iter.pdf"))
savefig(rd_iter_plot, joinpath(FIGS_PATH, "4-constrained-ls-rd-iter.pdf"))
savefig(obj_val_iter_plot, joinpath(FIGS_PATH, "4-constrained-ls-obj-val-iter.pdf"))
