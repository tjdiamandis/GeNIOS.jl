using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using StatsPlots: groupedbar
using IterativeSolvers, LinearMaps
using COSMO, OSQP
using JuMP, MosekTools
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const SAVEPATH = joinpath(@__DIR__, "saved", "4-constrained-ls-compare")
const FIGS_PATH = joinpath(@__DIR__, "figures")

const RAN_TRIALS = false

function run_trial(n::Int)
    GC.gc()
    BLAS.set_num_threads(Sys.CPU_THREADS)

    filename = "compare-$n.jld2"
    savefile = joinpath(SAVEPATH, filename)
    m = 2n

    Ad, b = get_augmented_data(m, n, DATAFILE)
    P, q, A, l, u = construct_problem_constrained_ls(Ad, b)
    GC.gc()

    A_sp = spdiagm(ones(n))
    P_sp = sparse(P)

    # OSQP
    GC.gc()
    osqp_model = OSQP.Model()
    OSQP.setup!(
        osqp_model; P=P_sp, q=q, A=A_sp, l=l, u=u, 
        eps_abs=1e-4, eps_rel=1e-4, verbose=false, time_limit=1000,
    )
    result_osqp = OSQP.solve!(osqp_model)
    @info "Finished OSQP"

    # COSMO
    GC.gc()
    model_cosmo_indirect = COSMO.Model()
    cs1 = COSMO.Constraint(spdiagm(ones(n)), zeros(n), COSMO.Box(Vector(l), u))
    settings = COSMO.Settings(
        kkt_solver=CGIndirectKKTSolver, 
        verbose=false,
        verbose_timing=true,
        eps_abs=1e-4,
        eps_rel=1e-4,
        time_limit=1000,
    )
    assemble!(model_cosmo_indirect, P, q, cs1, settings=settings)
    result_cosmo_indirect = COSMO.optimize!(model_cosmo_indirect)
    @info "Finished COSMO indirect"

    # COSMO
    GC.gc()
    model_cosmo_direct = COSMO.Model()
    settings = COSMO.Settings(
        kkt_solver=CholmodKKTSolver,
        verbose=false,
        verbose_timing = true,
        eps_abs=1e-4,
        eps_rel=1e-4,
        time_limit=1000,
    )
    assemble!(model_cosmo_direct, P, q, cs1, settings=settings)
    result_cosmo_direct = COSMO.optimize!(model_cosmo_direct)
    @info "Finished COSMO direct"

    # GeNIOS solve
    # With everything
    GC.gc()
    solver = GeNIOS.QPSolver(P, q, A, l, u)
    options = GeNIOS.SolverOptions(
        verbose=false,
        precondition=true,
        num_threads=Sys.CPU_THREADS,
        norm_type=Inf,
        sketch_update_iter=5000,
        max_time_sec=1000.
    )
    result = solve!(solver; options=options)
    @info "Finished GeNIOS"

    # No preconditioner
    GC.gc()
    solver_npc = GeNIOS.QPSolver(P, q, A, l, u)
    options = GeNIOS.SolverOptions(
        verbose=false,
        precondition=false,
        num_threads=Sys.CPU_THREADS,
        norm_type=Inf,
        max_time_sec=1000.
    )
    result_npc = solve!(solver_npc; options=options)
    @info "Finished GeNIOS (no pc)"

    # Mosek solve
    model = Model(Mosek.Optimizer)
    @variable(model, x[1:n])
    @objective(model, Min, 0.5*sum(P[i,j]*x[i]*x[j] for i in 1:n, j in 1:n) + sum(q[i]*x[i] for i in 1:n))
    @constraint(model, 0.0 .≤ x)
    @constraint(model, x .≤ 1.0)
    set_silent(model)
    set_time_limit_sec(model, 1000.0)
    optimize!(model)
    result_mosek = solution_summary(model)
    @info "Finished Mosek"

    save(savefile, 
        "result", result,
        "result_npc", result_npc,
        "result_cosmo_indirect", result_cosmo_indirect,
        "result_cosmo_direct", result_cosmo_direct,
        "result_osqp", result_osqp,
        "result_mosek", result_mosek,
    )
    
    GC.gc()
    return nothing
end

# n = 100 is just for compilation
ns = [250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
if !RAN_TRIALS
    run_trial(100)
    for n in ns
        println()
        @info "--- Starting n=$n ---"
        run_trial(n)
        @info "Finished with n=$n"
    end
end

function get_logs(n)
    savefile = joinpath(SAVEPATH, "compare-$n.jld2")
    r, r_npc, rc_indirect, rc_direct, r_osqp, r_mosek = load(savefile,
        "result", "result_npc", "result_cosmo_indirect", "result_cosmo_direct", "result_osqp", "result_mosek"
    )
    
    println(" --- n = $n --- ")
    for (x, y) in zip(
            ("GeNIOS", "GeNIOS (no pc)", "COSMO (indirect)", "COSMO (direct)", "OSQP", "Mosek"),
            (r, r_npc, rc_indirect, rc_direct, r_osqp.info)
        )
        println("$x: $(y.status)")
    end
    println("Mosek: $(r_mosek.termination_status)")
    
    return r, r_npc, rc_indirect, rc_direct, r_osqp, r_mosek
end

function get_timing(r, r_npc, rc_indirect, rc_direct, r_osqp, r_mosek)
    setup_times = [
        r.log.setup_time, 
        r_npc.log.setup_time, 
        rc_indirect.times.setup_time, 
        rc_direct.times.setup_time,
        r_osqp.info.setup_time,
        0.0,
    ]
    solve_times = [
        r.log.solve_time, 
        r_npc.log.solve_time, 
        rc_indirect.times.iter_time + rc_indirect.times.factor_update_time,
        rc_direct.times.iter_time + rc_direct.times.factor_update_time,
        r_osqp.info.solve_time,
        r_mosek.solve_time,
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
    ns, 
    timings, 
    yaxis=:log, 
    xaxis=:log,
    label=[
        "GeNIOS" "GeNIOS (no pc)" "COSMO (indirect)" "COSMO (direct)" "OSQP" "Mosek"
    ], 
    xlabel=L"Problem size $n$", 
    ylabel="Total solve time (s)", 
    legend=:bottomright,
    lw=2,
    markershape=[:circle :diamond :circle :cross :cross :square],
    color=[:coral :coral :mediumblue :mediumblue :black :purple],
    dpi=300,
    yticks=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    minorgrid=true,
)
savefig(timing_plt, joinpath(FIGS_PATH, "4-constrained-ls-compare-timing.pdf"))

inds_n = [2, 5, 7]
# inds_trial = [1, 3, 5, 6]
# names_trial = ["GeNIOS", "GeNIOS (exact)", "COSMO (indirect)", "COSMO (direct)"]
names = [
    "GeNIOS", "GeNIOS (no pc)", "GeNIOS (exact)", "GeNIOS (exact, no pc)", "COSMO (indirect)", "COSMO (direct)"
]

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
        ylims = (1e-4, 2e3),
        xticks=(1:length(inds_n), [L"$n = %$n$" for n in ns[inds_n]]),
        title=names[ind],
        titlefontsize=18,
        labelfontsize=16,
        tickfontsize=10,
        minorgrid=true,
    )
    for ind in 1:length(names)-1
]
setup_solve_plt = plot(
    plts..., 
    dpi=300,
    size=(1600, 900),
    leftmargin=10Plots.mm,
    bottommargin=10Plots.mm,
    topmargin=3Plots.mm,
)
savefig(setup_solve_plt, joinpath(FIGS_PATH, "4-constrained-ls-compare-setup-solve.pdf"))
