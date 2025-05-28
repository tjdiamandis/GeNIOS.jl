using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2
using COSMO, OSQP
using JuMP, MosekTools
using CSV, DataFrames, Statistics
include(joinpath(@__DIR__, "utils.jl"))
using GeNIOS

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE_DENSE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const DATAFILE_SPARSE_2 = joinpath(DATAPATH, "news20.jld2")
const SAVEPATH = joinpath(@__DIR__, "saved")
const SAVEFILE = "1-elastic-net-compare-may2025"
const FIGS_PATH = joinpath(@__DIR__, "figures")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA_SPARSE = false
const RAN_TRIALS = false


function run_trial(; type, m=10_000, n=20_000)
    if type == "sparse"
        # real-sim dataset
        A, b = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE, dataset_id=1578)
    elseif type == "sparse2"
        # news20 dataset
        A, b = load_sparse_data(file=DATAFILE_SPARSE_2, have_data=HAVE_DATA_SPARSE, dataset_id=1594)
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
    solve!(
        GeNIOS.ElasticNetSolver(λ1, λ2, A, b); 
        options=GeNIOS.SolverOptions(
            use_dual_gap=true,
            max_iters=2,
            verbose=false, 
            precondition=true
    ))

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
    time_genios = result.log.solve_time + result.log.setup_time



    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
    set_optimizer(model, OSQP.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "verbose", false)
    set_optimizer_attribute(model, "time_limit", 1800.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.:OPTIMAL && @warn "OSQP did not solve the problem"
    time_osqp = solve_time(model)
    @show time_osqp


    # COSMO (indirect)
    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
    set_optimizer(model, COSMO.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "kkt_solver", CGIndirectKKTSolver)
    set_optimizer_attribute(model, "verbose", false)
    set_optimizer_attribute(model, "time_limit", 1800.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "COSMO (indirect) did not solve the problem"
    time_cosmo_indirect = solve_time(model)
    @show time_cosmo_indirect


    # COSMO (direct)
    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
    set_optimizer(model, COSMO.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "verbose", false)
    set_optimizer_attribute(model, "time_limit", 1800.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "COSMO (direct) did not solve the problem"
    time_cosmo_direct = solve_time(model)
    @show time_cosmo_direct

    # Mosek
    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
    set_optimizer(model, Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-4)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-4)
    set_optimizer_attribute(model, "time_limit", 1800.0)
    GC.gc()
    optimize!(model)


    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    save(savefile, 
        "time_genios", time_genios,
        "time_osqp", time_osqp,
        "time_cosmo_indirect", time_cosmo_indirect,
        "time_cosmo_direct", time_cosmo_direct,
        "time_mosek", time_mosek
    )

    return nothing
end


if !RAN_TRIALS
    for type in ["sparse", "sparse2", "dense"]
        run_trial(type=type)
        @info "Finished with type=$type"
    end
end
@info "Finished with all trials"


println("\\begin{tabular}{@{}lrrrrr@{}}")
println("\\toprule")
println("Dataset & GeNIOS & OSQP & COSMO (indirect) & COSMO (direct) & Mosek \\\\")
println("\\midrule")

for type in ["sparse", "sparse2", "dense"]
    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    time_genios, time_osqp, time_cosmo_indirect, time_cosmo_direct, time_mosek = 
        load(savefile, "time_genios", "time_osqp", "time_cosmo_indirect", "time_cosmo_direct", "time_mosek")
    
    println("$type & $(@sprintf("%.3f", time_genios)) & $(@sprintf("%.3f", time_osqp)) & $(@sprintf("%.3f", time_cosmo_indirect)) & $(@sprintf("%.3f", time_cosmo_direct)) & $(@sprintf("%.3f", time_mosek)) \\\\")
end

println("\\bottomrule")
println("\\end{tabular}")
