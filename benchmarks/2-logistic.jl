using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2
using CSV, DataFrames, Statistics, JLD2
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

DATAPATH = joinpath(@__DIR__, "data")
# DATAFILE = joinpath(DATAPATH, "YearPredictionMSD.txt")
DATAFILE = joinpath(DATAPATH, "year-pred.jld2")
SAVEPATH = joinpath(@__DIR__, "saved")
SAVEFILE = joinpath(SAVEPATH, "2-logistic.jld2")
FIGS_PATH = joinpath(@__DIR__, "figures")


## Construct the problem data
m, n = 1_000, 5_000
BLAS.set_num_threads(Sys.CPU_THREADS)

# Set this to false if you have not yet downloaded the real-sim dataset
HAVE_DATA = false

if !HAVE_DATA
    dataset = OpenML.load(44027)

    A_full = sparse(Tables.matrix(dataset)[:,1:end-1])
    b_full = dataset.year
    save(DATAFILE, "A_full", A_full, "b_full", b_full)
else
    A_full, b_full = load(DATAFILE, "A_full", "b_full")
end
A_full .= A_full .- mean(A_full; dims=1)
A_full .= A_full ./ std(A_full, dims=1)
b_full .= b_full .- mean(b_full)
b_full .= b_full ./ std(b_full)

A_non_augmented = A_full[1:m, :]
b = b_full[1:m]

σ = 8
A = zeros(m, n)
gauss_fourier_features!(A, A_non_augmented, σ)
GC.gc()

# Reguarlization parameters
λ1_max = norm(A'*b, Inf)
λ1 = 0.05*λ1_max
λ2 = 0.0

## Solving the Problem
# For compilation
solve!(GeNIOS.LogisticSolver(λ1, λ2, A, b); options=GeNIOS.SolverOptions(max_iters=2))

# With everything
solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
options = GeNIOS.SolverOptions(
    relax=true,
    α=1.6,
    use_dual_gap=true,
    precondition=true,
    num_threads=Sys.CPU_THREADS,
    max_iters=2000,
)
GC.gc()
result = solve!(solver; options=options)

# No preconditioner
solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
options = GeNIOS.SolverOptions(
    relax=true,
    α=1.6,
    use_dual_gap=true,
    verbose=true,
    precondition=false,
    num_threads=Sys.CPU_THREADS,
    max_iters=2000,
)
GC.gc()
result_npc = solve!(solver; options=options)

# Exact solve
solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
options = GeNIOS.SolverOptions(
    relax=true,
    α=1.6,
    use_dual_gap=true,
    verbose=true,
    precondition=true,
    num_threads=Sys.CPU_THREADS,
    linsys_max_tol=1e-8,
    max_iters=2000,
)
GC.gc()
result_exact = solve!(solver; options=options)

# Exact solve, no pc
solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
options = GeNIOS.SolverOptions(
    relax=true,
    α=1.6,
    use_dual_gap=true,
    verbose=true,
    precondition=false,
    num_threads=Sys.CPU_THREADS,
    linsys_max_tol=1e-8,
    max_iters=2000,
)
GC.gc()
result_exact_npc = solve!(solver; options=options)

# High precision solve
solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
options = GeNIOS.SolverOptions(
    relax=true,
    α=1.6,
    use_dual_gap=true,
    dual_gap_tol = sqrt(eps()),
    verbose=true,
    precondition=true,
    num_threads=Sys.CPU_THREADS,
    max_iters=10_000
)
GC.gc()
result_high_precision = solve!(solver; options=options)


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
log = result.log
log_npc = result_npc.log
log_exact = result_exact.log
log_exact_npc = result_exact_npc.log
log_high_precision = result_high_precision.log
pstar = log_high_precision.obj_val[end]

# Printout timings
print_timing("GeNIOS", log)
print_timing("GeNIOS (no pc)", log_npc)
print_timing("ADMM (exact)", log_exact)
print_timing("ADMM (exact, no pc)", log_exact_npc)

names = ["GeNIOS", "GeNIOS (no pc)", "`ADMM'", "`ADMM' (no pc)"]
logs = [log, log_npc, log_exact, log_exact_npc]
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

savefig(dual_gap_iter_plt, joinpath(FIGS_PATH, "2-logistic-dual-gap-iter.pdf"))
savefig(rp_iter_plot, joinpath(FIGS_PATH, "2-logistic-rp-iter.pdf"))
savefig(rd_iter_plot, joinpath(FIGS_PATH, "2-logistic-rd-iter.pdf"))
savefig(obj_val_iter_plot, joinpath(FIGS_PATH, "2-logistic-obj-val-iter.pdf"))
