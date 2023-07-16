using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2
using IterativeSolvers, LinearMaps
using Optim
include(joinpath(@__DIR__, "utils.jl"))


const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE_DENSE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const SAVEPATH = joinpath(@__DIR__, "saved", "2-logistic")
const FIGS_PATH = joinpath(@__DIR__, "figures")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA_SPARSE = true
const RAN_TRIALS = true


function build_genios_conic_model(A, b, λ1)
    N, n = size(A)
    nn = 2n + 5N
    P = spzeros(nn, nn)
    q = zeros(nn)
    q[n+1:2n] .= λ1
    q[2n+1:2n+N] .= 1.0

    is_1 = vcat(1:n, n+1:2n, 1:n, n+1:2n)
    js_1 = vcat(1:n, 1:n, n+1:2n, n+1:2n)
    vs_1 = vcat(-ones(n), ones(n), ones(n), ones(n))
    R1 = sparse(is_1, js_1, vs_1, 2n, nn)
    # In = sparse(1.0I, n, n)
    # R1 = spzeros(2n, nn)
    # R1[1:n, 1:n] .= -In
    # R1[n+1:2n, 1:n] .= In
    # R1[1:n, n+1:2n] .= In
    # R1[n+1:2n, n+1:2n] .= In
    K1 = GeNIOS.IntervalCone(zeros(2n), Inf*ones(2n))

    IN = sparse(1.0I, N, N)

    is_2 = vcat(1:N, 1:N)
    js_2 = vcat(2n+2N+1:2n+3N, 2n+4N+1:2n+5N)
    vs_2 = ones(2N)
    R2 = sparse(is_2, js_2, vs_2, N, nn)
    # R2 = spzeros(N, nn)
    # R2[1:N, 2n+2N+1:2n+3N] .= IN
    # R2[1:N, 2n+4N+1:2n+5N] .= IN
    K2 = GeNIOS.IntervalCone(-Inf*ones(N), ones(N))

    i5 = 1:N
    j5 = 2n+N+1:2n+2N
    v5 = ones(N)
    R5 = sparse(i5, j5, v5, N, nn)
    # R5 = spzeros(N, nn)
    # R5[1:N, 2n+N+1:2n+2N] .= IN
    K5 = GeNIOS.IntervalCone(ones(N), ones(N))

    i6 = 1:N
    j6 = 2n+3N+1:2n+4N
    v6 = ones(N)
    R6 = sparse(i6, j6, v6, N, nn)
    # R6 = spzeros(N, nn)
    # R6[1:N, 2n+3N+1:2n+4N] .= IN
    K6 = GeNIOS.IntervalCone(ones(N), ones(N))

    i3 = vcat(1:3:3N, 2:3:3N, 3:3:3N)
    j3 = vcat(2n+1:2n+N, 2n+N+1:2n+2N, 2n+2N+1:2n+3N)
    v3 = vcat(-ones(N), ones(N), ones(N))
    for (i, j, v) in zip(findnz(A)...)
        push!(i3, 3(i-1)+1)
        push!(j3, j)
        push!(v3, v)
    end
    R3 = sparse(i3, j3, v3, 3N, nn)
    # R3 = spzeros(3N, nn)
    # R3[1:3:end, 1:n] .= A
    # R3[1:3:end, 2n+1:2n+N] .= -IN
    # R3[2:3:end, 2n+N+1:2n+2N] .= IN
    # R3[3:3:end, 2n+2N+1:2n+3N] .= IN
    K3 = fill(GeNIOS.ExponentialCone(), N)

    i4 = vcat(1:3:3N, 2:3:3N, 3:3:3N)
    j4 = vcat(2n+1:2n+N, 2n+3N+1:2n+4N, 2n+4N+1:2n+5N)
    v4 = vcat(-ones(N), ones(N), ones(N))
    R4 = sparse(i4, j4, v4, 3N, nn)
    # R4 = spzeros(3N, nn)
    # R4[1:3:end, 2n+1:2n+N] .= -IN
    # R4[2:3:end, 2n+3N+1:2n+4N] .= IN
    # R4[3:3:end, 2n+4N+1:2n+5N] .= IN
    K4 = fill(GeNIOS.ExponentialCone(), N)

    M = vcat(R1, R2, R3, R4, R5, R6)
    c = zeros(size(M, 1))
    K = GeNIOS.ProductCone(vcat(K1, K2, K3, K4, K5, K6))

    return GeNIOS.ConicSolver(P, q, K, M, c)
end



function run_trial()
    BLAS.set_num_threads(Sys.CPU_THREADS)
    A_full, b_full = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE)
    N, n = size(A_full)

    # Reguarlization parameters
    λ1_max = norm(A_full'*b_full, Inf)
    λ1 = 0.1*λ1_max
    λ2 = 0.0

    A = Diagonal(b_full) * A_full
    # N = 10_000
    # A = A[1:N, :]
    b = zeros(N)

    ## Solving the Problem
    # For compilation
    solve!(GeNIOS.LogisticSolver(λ1, λ2, A, b); options=GeNIOS.SolverOptions(max_iters=2, use_dual_gap=true, verbose=false))

    # With everything
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-4,
        precondition=true,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=20*60.
    )
    GC.gc()
    result = solve!(solver; options=options)

    # No preconditioner
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-4,
        verbose=true,
        precondition=false,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=20*60.
    )
    GC.gc()
    result_npc = solve!(solver; options=options)

    # Exact solve
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-4,
        verbose=true,
        precondition=true,
        linsys_max_tol=1e-8,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=20*60.
    )
    GC.gc()
    result_exact = solve!(solver; options=options)

    # Exact solve, no pc
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol=1e-4,
        verbose=true,
        precondition=false,
        linsys_max_tol=1e-8,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=20*60.
    )
    GC.gc()
    result_exact_npc = solve!(solver; options=options)

    # High precision solve
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
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


    # LBFGS solve
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        use_dual_gap=true,
        dual_gap_tol = 1e-4,
        verbose=true,
        precondition=false,
        num_threads=Sys.CPU_THREADS,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
    )
    GC.gc()
    # compile then solve
    solve!(solver; options=GeNIOS.SolverOptions(max_iters=2, verbose=false), use_lbfgs_ml=true)
    result_lbfgs = solve!(solver; options=options, use_lbfgs_ml=true)


    # Conic solve
    solver = build_genios_conic_model(A, b, λ1)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        eps_abs=1e-4,
        eps_rel=1e-4,
        verbose=true,
        precondition=false,
        num_threads=Sys.CPU_THREADS,
        ρ0=1.0,
        rho_update_iter=1000,
        max_iters=5000,
    )
    GC.gc()
    # compile then solve 
    solve!(solver; options=GeNIOS.SolverOptions(max_iters=2, precondition=false, verbose=false))
    result_conic = solve!(solver; options=options)


    savefile = joinpath(SAVEPATH, "2-logistic-genios.jld2")
    save(savefile, 
        "result", result,
        "result_npc", result_npc,
        "result_exact", result_exact,
        "result_exact_npc", result_exact_npc,
        "result_high_precision", result_high_precision,
        "result_lbfgs", result_lbfgs,
        "result_conic", result_conic,
    )

    return nothing
end

!RAN_TRIALS && run_trial()

savefile = joinpath(SAVEPATH, "2-logistic-genios.jld2")
result, result_npc, result_exact, result_exact_npc, result_high_precision, result_lbfgs, result_conic = 
    load(savefile, 
        "result", "result_npc", "result_exact", "result_exact_npc", "result_high_precision", "result_lbfgs", "result_conic",
    )
any(!isnothing(x) && x.status != :OPTIMAL for x in (
    result,
    result_npc,
    result_exact,
    result_exact_npc,
    result_high_precision
)) && @warn "Some problems not solved! for genios"

log = result.log
log_npc = result_npc.log
log_exact = result_exact.log
log_exact_npc = result_exact_npc.log
log_high_precision = result_high_precision.log
pstar = log_high_precision.obj_val[end]
log_lbfgs = result_lbfgs.log
log_conic = result_conic.log

names = ["GeNIOS", "GeNIOS (no pc)", "GeNIOS (exact)", "GeNIOS (no pc, exact)", "ADMM LBFGS", "Conic"]
logs = [log, log_npc, log_exact, log_exact_npc, log_lbfgs, log_conic]
@info "\n\n ----- TIMING TABLE -----\n"
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
add_to_plot!(dual_gap_iter_plt, log_lbfgs.iter_time, log_lbfgs.dual_gap, "LBFGS", :black);
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
add_to_plot!(rp_iter_plot, log_lbfgs.iter_time, log_lbfgs.rp, "LBFGS", :black);
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
add_to_plot!(rd_iter_plot, log_lbfgs.iter_time, log_lbfgs.rd, "LBFGS", :black);
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
add_to_plot!(obj_val_iter_plot, log_lbfgs.iter_time, (log_lbfgs.obj_val .- pstar) ./ pstar, "LBFGS", :black);
obj_val_iter_plot

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

savefig(dual_gap_iter_plt, joinpath(FIGS_PATH, "2-logistic-dual-gap-iter-genios.pdf"))
savefig(rp_iter_plot, joinpath(FIGS_PATH, "2-logistic-rp-iter-genios.pdf"))
savefig(rd_iter_plot, joinpath(FIGS_PATH, "2-logistic-rd-iter-genios.pdf"))
savefig(obj_val_iter_plot, joinpath(FIGS_PATH, "2-logistic-obj-val-iter-genios.pdf"))
savefig(high_precision_plt, joinpath(FIGS_PATH, "2-logistic-high-precision-genios.pdf"))