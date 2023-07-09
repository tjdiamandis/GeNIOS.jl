using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using IterativeSolvers, LinearMaps
using COSMO
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const SAVEPATH = joinpath(@__DIR__, "saved", "4-constrained-ls-compare")
const FIGS_PATH = joinpath(@__DIR__, "figures")

const RAN_TRIALS = true

function run_trial(n::Int)
    GC.gc()
    BLAS.set_num_threads(1)

    filename = "compare-$n.jld2"
    savefile = joinpath(SAVEPATH, filename)
    m = 2n
    P, q, A, l, u = construct_problem_constrained_ls(get_augmented_data(m, n, DATAFILE)...)

    model_cosmo_indirect = COSMO.Model()
    cs1 = COSMO.Constraint(spdiagm(ones(n)), zeros(n), COSMO.Box(Vector(l), u))
    settings = COSMO.Settings(
        kkt_solver=CGIndirectKKTSolver, 
        verbose=false,
        verbose_timing = true
    )
    assemble!(model_cosmo_indirect, P, q, cs1, settings=settings)
    GC.gc()
    result_cosmo_indirect = COSMO.optimize!(model_cosmo_indirect)

    model_cosmo_direct = COSMO.Model()
    settings = COSMO.Settings(
        kkt_solver=CholmodKKTSolver,
        verbose=false,
        verbose_timing = true,
    )
    assemble!(model_cosmo_direct, P, q, cs1, settings=settings)
    GC.gc()
    result_cosmo_direct = COSMO.optimize!(model_cosmo_direct)

    # GeNIOS solve
    # With everything
    solver = GeNIOS.QPSolver(P, q, A, l, u)
    options = GeNIOS.SolverOptions(
        verbose=false,
        precondition=true,
        num_threads=1,
    )
    GC.gc()
    result = solve!(solver; options=options)

    # No preconditioner
    solver = GeNIOS.QPSolver(P, q, A, l, u)
    options = GeNIOS.SolverOptions(
        verbose=false,
        precondition=false,
        num_threads=1,
    )
    GC.gc()
    result_npc = solve!(solver; options=options)

    # Exact solve
    solver = GeNIOS.QPSolver(P, q, A, l, u)
    options = GeNIOS.SolverOptions(
        verbose=false,
        precondition=true,
        num_threads=1,
        linsys_max_tol=1e-8
    )
    GC.gc()
    result_exact = solve!(solver; options=options)

    # Exact solve, no pc
    solver = GeNIOS.QPSolver(P, q, A, l, u)
    options = GeNIOS.SolverOptions(
        verbose=false,
        precondition=false,
        num_threads=1,
        linsys_max_tol=1e-8,
    )
    GC.gc()
    result_exact_npc = solve!(solver; options=options)

    save(savefile, 
        "result", result,
        "result_npc", result_npc,
        "result_exact", result_exact,
        "result_exact_npc", result_exact_npc,
        "result_cosmo_indirect", result_cosmo_indirect,
        "result_cosmo_direct", result_cosmo_direct,
    )
    
    GC.gc()
    return nothing
end

# n = 100 is just for compilation
ns = [100, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
if !RAN_TRIALS
    for n in ns
        run_trial(n)
        @info "Finished with n=$n"
    end
end

function get_logs(n)
    savefile = joinpath(SAVEPATH, "compare-$n.jld2")
    r, r_npc, r_exact, r_exact_npc, rc_indirect, rc_direct = 
        load(savefile, 
            "result", "result_npc", "result_exact", "result_exact_npc",
            "result_cosmo_indirect", "result_cosmo_direct"
        )
    
    if any(x.status ∉ (:OPTIMAL, :Solved) for x in (r, r_npc, r_exact, r_exact_npc, rc_indirect, rc_direct))
        @warn "n = $n: Some problems not solved!"
    end
    
    return r, r_npc, r_exact, r_exact_npc, rc_indirect, rc_direct
end

function get_timing(r, r_npc, r_exact, r_exact_npc, rc_indirect, rc_direct)
    setup_times = [
        r.log.setup_time, 
        r_npc.log.setup_time, 
        r_exact.log.setup_time, 
        r_exact_npc.log.setup_time, 
        rc_indirect.times.setup_time, 
        rc_direct.times.setup_time
    ]
    solve_times = [
        r.log.solve_time, 
        r_npc.log.solve_time, 
        r_exact.log.solve_time, 
        r_exact_npc.log.solve_time, 
        rc_indirect.times.iter_time + rc_indirect.times.factor_update_time,
        rc_direct.times.iter_time + rc_direct.times.factor_update_time
    ]

    return setup_times, solve_times
end

timings = zeros(length(ns), 6)
setup_times = zeros(length(ns), 6)
solve_times = zeros(length(ns), 6)
for (i, n) in enumerate(ns)
    logs = get_logs(n)
    setup_time, solve_time = get_timing(logs...)
    timings[i,:] .= setup_time .+ solve_time
    setup_times[i,:] .= setup_time
    solve_times[i,:] .= solve_time
end

timing_plt = plot(
    ns[3:end].^2, 
    timings[3:end,:], 
    yaxis=:log, 
    xaxis=:log,
    label=[
        "GeNIOS" "GeNIOS (no pc)" "GeNIOS (exact)" "GeNIOS (exact, no pc)" "COSMO (indirect)" "COSMO (direct)"
    ], 
    xlabel=L"Elements in (dense) matrix $P$ $(n^2)$", 
    ylabel="Total solve time (s)", 
    legend=:bottomright,
    lw=2,
    markershape=[:circle :diamond :utriangle :star5 :square :cross],
    dpi=300,
    yticks=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
)
savefig(joinpath(FIGS_PATH, "4-constrained-ls-compare-timing.pdf"))

inds_n = [3, 5, 7]
# inds_trial = [1, 3, 5, 6]
# names_trial = ["GeNIOS", "GeNIOS (exact)", "COSMO (indirect)", "COSMO (direct)"]
names = [
    "GeNIOS", "GeNIOS (no pc)", "GeNIOS (exact)", "GeNIOS (exact, no pc)", "COSMO (indirect)", "COSMO (direct)"
]

using StatsPlots: groupedbar
plts = [
    groupedbar(
        [setup_times[inds_n, ind] solve_times[inds_n, ind]],
        label=["setup" "solve"],
        bar_width=0.8,
        xlabel= ind > 3 ? L"Matrix $P$ side dimension $n$" : "",
        ylabel= ind % 3 == 1 ? "Time (s)" : "",
        legend=:topleft,
        dpi=300,
        yaxis=:log,
        yticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        ylims = (1e-4, 1e3),
        xticks=(1:length(inds_n), [L"$n = %$n$" for n in ns[inds_n]]),
        title=names[ind],
        titlefontsize=18,
        labelfontsize=16,
        tickfontsize=10,
    )
    for ind in 1:length(names)
]
setup_slove_plt = plot(
    plts..., 
    dpi=300,
    size=(1600, 900),
    leftmargin=10Plots.mm,
    bottommargin=10Plots.mm,
    topmargin=3Plots.mm,
)
savefig(joinpath(FIGS_PATH, "4-constrained-ls-compare-setup-solve.pdf"))