using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
# using OpenML, Tables, JLD2
using CSV, DataFrames, Statistics, JLD2
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

DATAPATH = joinpath(@__DIR__, "data")
DATAFILE = joinpath(DATAPATH, "YearPredictionMSD.txt")
SAVEPATH = joinpath(@__DIR__, "saved")
SAVEFILE = joinpath(SAVEPATH, "4-constrained-ls.jld2")
FIGS_PATH = joinpath(@__DIR__, "figures")


## Construct the problem data
m, n = 1_000, 5_000
BLAS.set_num_threads(Sys.CPU_THREADS)

file = CSV.read(DATAFILE, DataFrame)
M = Matrix(file[1:m,:])
size(M, 1)
M .= M .- sum(M, dims=1) ./ size(M, 1)
M .= M ./ std(M, dims=1)

A_non_augmented = M[:, 2:end]
b = M[:, 1]

σ = 8
Ad = zeros(m, n)
gauss_fourier_features!(Ad, A_non_augmented, σ)
GC.gc()


## Setup problem
P = blockdiag(spzeros(n, n), sparse(I, m, m))
q = spzeros(n + m)
A = [
    Ad                   -sparse(I, m, m);
    sparse(I, n, n)     spzeros(n, m)
]
l = [b; spzeros(n)]
u = [b; ones(n)]


# compile
solve!(GeNIOS.QPSolver(P, q, A, l, u); options=GeNIOS.SolverOptions(max_iters=2))

# solve
# With everything
solver = GeNIOS.QPSolver(P, q, A, l, u)
options = GeNIOS.SolverOptions(
    precondition=true,
    num_threads=Sys.CPU_THREADS,
)
GC.gc()
result = solve!(solver; options=options)

# No preconditioner
solver = GeNIOS.QPSolver(P, q, A, l, u)
options = GeNIOS.SolverOptions(
    precondition=false,
    num_threads=Sys.CPU_THREADS,
)
GC.gc()
result_npc = solve!(solver; options=options)

# Exact solve
solver = GeNIOS.QPSolver(P, q, A, l, u)
options = GeNIOS.SolverOptions(
    precondition=true,
    num_threads=Sys.CPU_THREADS,
    linsys_max_tol=1e-8
)
GC.gc()
result_exact = solve!(solver; options=options)

# Exact solve, no pc
solver = GeNIOS.QPSolver(P, q, A, l, u)
options = GeNIOS.SolverOptions(
    precondition=false,
    num_threads=Sys.CPU_THREADS,
    linsys_max_tol=1e-8
)
GC.gc()
result_exact_npc = solve!(solver; options=options)

# High precision solve
solver = GeNIOS.QPSolver(P, q, A, l, u)
options = GeNIOS.SolverOptions(
    eps_abs=1e-8,
    precondition=true,
    num_threads=Sys.CPU_THREADS,
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
log_pc = result.log
log_npc = result_npc.log
log_exact = result_exact.log
log_exact_npc = result_exact_npc.log
log_high_precision = result_high_precision.log
pstar = log_high_precision.obj_val[end]

# Printout timings
print_timing("GeNIOS", log_pc)
print_timing("GeNIOS (no pc)", log_npc)
print_timing("ADMM (exact)", log_exact)
print_timing("ADMM (exact, no pc)", log_exact_npc)



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
add_to_plot!(obj_val_iter_plot, log_pc.iter_time[2:end], (log_pc.obj_val[2:end] .- pstar) ./ pstar, "GeNIOS", :coral);
add_to_plot!(obj_val_iter_plot, log_npc.iter_time[2:end], (log_npc.obj_val[2:end] .- pstar) ./ pstar, "No PC", :purple);
add_to_plot!(obj_val_iter_plot, log_exact.iter_time[2:end], (log_exact.obj_val[2:end] .- pstar) ./ pstar, "ADMM (pc)", :red);
add_to_plot!(obj_val_iter_plot, log_exact_npc.iter_time[2:end], (log_exact_npc.obj_val[2:end] .- pstar) ./ pstar, "ADMM (no pc)", :mediumblue);
obj_val_iter_plot

savefig(rp_iter_plot, joinpath(FIGS_PATH, "4-constrained-ls-rp-iter.pdf"))
savefig(rd_iter_plot, joinpath(FIGS_PATH, "4-constrained-ls-rd-iter.pdf"))
savefig(obj_val_iter_plot, joinpath(FIGS_PATH, "4-constrained-ls-obj-val-iter.pdf"))
