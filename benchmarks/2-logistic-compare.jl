using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2
using COSMO, OSQP, GeNIADMM
using IterativeSolvers, LinearMaps
using JuMP, MosekTools
using CSV, DataFrames, Statistics
include(joinpath(@__DIR__, "utils.jl"))
using GeNIOS

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const DATAFILE_SPARSE_2 = joinpath(DATAPATH, "news20.jld2")
const SAVEPATH = joinpath(@__DIR__, "saved")
const SAVEFILE = "2-logistic-compare-may2025"
const FIGS_PATH = joinpath(@__DIR__, "figures")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA_SPARSE = true
const HAVE_DATA_SPARSE2 = true
const RAN_TRIALS = false
const TEST_MODE = false

function run_trial(; type, n=100)
    if type == "test"
        # real-sim dataset
        A_full, b_full = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE, dataset_id=1578)
        A_full = A_full[1:n, 1:2n]
        b_full = b_full[1:n]
    elseif type == "sparse"
        # real-sim dataset
        A_full, b_full = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE, dataset_id=1578)
    elseif type == "sparse2"
        # news20 dataset
        A_full, b_full = load_sparse_data(file=DATAFILE_SPARSE_2, have_data=HAVE_DATA_SPARSE2, dataset_id=1594)
        # make into two classes:
        b_full[b_full .<= 10] .= -1.0
        b_full[b_full .> 10] .= 1.0
    else
        error("Unknown type: $type")
    end
    @info "Starting type = $type"

    # Reguarlization parameters
    λ1_max = norm(A_full'*b_full, Inf)
    λ1 = 0.05*λ1_max
    λ2 = 0.0

    A = Diagonal(b_full) * A_full
    b = zeros(size(A, 1))

    # For compilation
    solve!(
        GeNIOS.LogisticSolver(λ1, λ2, A, b); 
        options=GeNIOS.SolverOptions(
            use_dual_gap=true,
            max_iters=2,
            verbose=false, 
            precondition=true
    ))

    # GeNIOS
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        eps_abs=1e-4,
        eps_rel=1e-4,
        verbose=false,
        precondition=true,
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=1800.
    )
    GC.gc()
    result = solve!(solver; options=options)
    time_genios = result.log.solve_time + result.log.setup_time
    @info "    GeNIOS: $(time_genios)"

    # COSMO (indirect)
    model = construct_jump_model_logistic(A, b, λ1)
    set_optimizer(model, COSMO.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "kkt_solver", CGIndirectKKTSolver)
    set_optimizer_attribute(model, "verbose", false)
    set_time_limit_sec(model, 1800.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "COSMO (indirect) did not solve the problem"
    time_cosmo_indirect = solve_time(model)
    @info "    COSMO (indirect): $(time_cosmo_indirect)"


    # COSMO (direct)
    model = construct_jump_model_logistic(A, b, λ1)
    set_optimizer(model, COSMO.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "verbose", false)
    set_time_limit_sec(model, 1800.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "COSMO (direct) did not solve the problem"
    time_cosmo_direct = solve_time(model)
    @info "    COSMO (direct): $(time_cosmo_direct)"

    # Mosek
    model = construct_jump_model_logistic(A, b, λ1)
    set_optimizer(model, Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-4)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-4)
    set_time_limit_sec(model, 1800.0)
    set_silent(model)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "Mosek did not solve the problem"
    time_mosek = solve_time(model)
    @info "    Mosek: $(time_mosek)"


    # FISTA
    prob = GeNIADMM.LogisticSolver(A, b, λ1)
    GC.gc()
    GeNIADMM.solve!(
        prob; indirect=true, relax=false, max_iters=1, tol=1e-4, logging=true,
        precondition=false, verbose=false, print_iter=100, agd_x_update=true,
        rho_update_iter=10_000, multithreaded=true
    )
    result_fista = GeNIADMM.solve!(
        prob; indirect=true, relax=false, max_iters=5_000, tol=1e-4, logging=true,
        precondition=false, verbose=false, print_iter=100, agd_x_update=true,
        rho_update_iter=10_000, multithreaded=true
    )
    time_fista = result_fista.log.solve_time + result_fista.log.setup_time
    @info "    FISTA: $(time_fista)"


    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    save(savefile, 
        "time_genios", time_genios,
        "time_cosmo_indirect", time_cosmo_indirect,
        "time_cosmo_direct", time_cosmo_direct,
        "time_mosek", time_mosek,
        "time_fista", time_fista
    )

    return nothing
end


if !RAN_TRIALS
    # compile
    run_trial(type="test")
    @info "Finished compiling"
    if !TEST_MODE
        types = ["sparse", "sparse2"]
        for type in types
            run_trial(type=type)
            @info "Finished with type=$type"
        end
    end
end
@info "Finished with all trials"


println("\\begin{tabular}{@{}lrrrrr@{}}")
println("\\toprule")
println("Dataset & GeNIOS & COSMO (indirect) & COSMO (direct) & Mosek & FISTA \\\\")
println("\\midrule")
for type in ["sparse", "sparse2"]
    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    time_genios, time_cosmo_indirect, time_cosmo_direct, time_mosek, time_fista = 
        load(savefile, "time_genios", "time_cosmo_indirect", "time_cosmo_direct", "time_mosek", "time_fista")
    
    println("$type & $(@sprintf("%.3f", time_genios)) & $(@sprintf("%.3f", time_cosmo_indirect)) & $(@sprintf("%.3f", time_cosmo_direct)) & $(@sprintf("%.3f", time_mosek)) & $(@sprintf("%.3f", time_fista)) \\\\")
end
println("\\bottomrule")
println("\\end{tabular}")
