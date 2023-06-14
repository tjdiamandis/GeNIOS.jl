using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2, Statistics
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

# DATAPATH = joinpath(@__DIR__, "data")
# DATAFILE = joinpath(DATAPATH, "year-pred.jld2")
SAVEPATH = joinpath(@__DIR__, "saved")
SAVEFILE = joinpath(SAVEPATH, "3-huber.jld2")
FIGS_PATH = joinpath(@__DIR__, "figures")


## Generating the problem data
Random.seed!(1)
BLAS.set_num_threads(Sys.CPU_THREADS)
N, n = 2_000, 4_000
A = randn(N, n)
A .-= sum(A, dims=1) ./ N
normalize!.(eachcol(A))
xstar = sprandn(n, 0.1)
b = A*xstar + 1e-3*randn(N)

# Add outliers
b += 10*collect(sprand(N, 0.05))
GC.gc()

# Reguarlization parameters
λ1_max = norm(A'*b, Inf)
λ1 = 0.05*λ1_max
λ2 = 0.0

f(x) = abs(x) <= 1 ? 0.5*x^2 : abs(x) - 0.5
df(x) = abs(x) <= 1 ? x : sign(x)
d2f(x) = abs(x) <= 1 ? 1 : 0


## Solve with MLSolver
# compile
solve!(GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b); options=GeNIOS.SolverOptions(max_iters=2))

solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b)
options = GeNIOS.SolverOptions(
    relax=false,
    use_dual_gap=false,
    verbose=true
)
GC.gc()
result_ml = solve!(solver; options=options)


## Solve with QPSolver
IN = sparse(I, N, N)
P = blockdiag(spzeros(n,n), 2*IN, spzeros(2N,2N))
q = vcat(zeros(n+N), 2*ones(2N))

Zn = spzeros(N,n)
ZN = spzeros(N,N)
Aqp = [A       -IN    -IN    IN
       Zn      ZN     IN     ZN;
       Zn      ZN     ZN     IN]
l = vcat(b, zeros(2N))
u = vcat(b, Inf*ones(2N))

# compile
solve!(GeNIOS.QPSolver(P, q, Aqp, l, u); options=GeNIOS.SolverOptions(max_iters=2))

solver = GeNIOS.QPSolver(P, q, Aqp, l, u)
options = GeNIOS.SolverOptions(
    relax=false,
    use_dual_gap=false,
    verbose=true
)
GC.gc()
result_qp = solve!(solver; options=options)

# save results
save(SAVEFILE,
    "result_ml", result_ml,
    "result_qp", result_qp
)


## Load data from save files
result_ml, result_qp = load(SAVEFILE, "result_ml", "result_qp")
any(x.status != :OPTIMAL for x in [result_ml, result_qp]) && @warn "Some problems not solved!"

log_ml = result_ml.log
log_qp = result_qp.log

# Print timings
print_timing("MLSolver", log_ml)
print_timing("QPSolver", log_qp)

# Plot things
# TODO: add custom convergence criterion so looking at the same thing
rp_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Primal Residual $\ell_2$ Norm",
    xlabel="Time (s)",
    legend=:topright,
)
add_to_plot!(rp_iter_plot, log_ml.iter_time, log_ml.rp, "MLSolver", :coral);
add_to_plot!(rp_iter_plot, log_qp.iter_time, log_qp.rp, "QPSolver", :purple);
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
add_to_plot!(rd_iter_plot, log_ml.iter_time, log_ml.rd, "MLSolver", :coral);
add_to_plot!(rd_iter_plot, log_qp.iter_time, log_qp.rd, "QPSolver", :purple);
rd_iter_plot

# Save figures
savefig(rp_iter_plot, joinpath(FIGS_PATH, "3-huber-rp-iter.pdf"))
savefig(rd_iter_plot, joinpath(FIGS_PATH, "3-huber-rd-iter.pdf"))
