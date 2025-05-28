using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using Statistics, JLD2
using IterativeSolvers, LinearMaps
using COSMO, OSQP
using MosekTools, JuMP
include(joinpath(@__DIR__, "utils.jl"))
using GeNIOS

const SAVEPATH = joinpath(@__DIR__, "saved", "5-portfolio")
const SAVEFILE = "5-portfolio-may2025"
const FIGS_PATH = joinpath(@__DIR__, "figures")
const RAN_TRIALS = false

# FOR QP SOLVER (custom operators)
# P = γ*(F*F' + Diagonal(d))
struct FastP
    F
    d
    γ
    vk
end
function LinearAlgebra.mul!(y::AbstractVector, P::FastP, x::AbstractVector)
    mul!(P.vk, P.F', x)
    mul!(y, P.F, P.vk)
    @. y += P.d*x
    @. y *= P.γ
    return nothing
end

# M = vcat(I, ones(1, n))
struct FastM 
    n::Int
end
Base.size(M::FastM) = (M.n+1, M.n)
Base.size(M::FastM, d::Int) = d <= 2 ? size(M)[d] : 1
function LinearAlgebra.mul!(y::AbstractVector, M::FastM, x::AbstractVector)
    y[1:M.n] .= x
    y[end] = sum(x)
    return nothing
end
LinearAlgebra.adjoint(M::FastM) = Adjoint{Float64, FastM}(M)
function LinearAlgebra.mul!(x::AbstractVector{T}, M::Adjoint{T, FastM}, y::AbstractVector{T}) where T <: Number
    @. x = y[1:M.parent.n] + y[end]
    return nothing
end
function LinearAlgebra.mul!(x::AbstractVector{T}, M::Adjoint{T, FastM}, y::AbstractVector{T}, α::T, β::T) where T <: Number
    @. x = α * ( y[1:M.parent.n] + y[end] ) + β * x
    return nothing
end

# FOR GENERIC SOLVER
# ∇²f(x) = γ(FFᵀ + D)
struct HessianMarkowitz{T, S <: AbstractMatrix{T}} <: HessianOperator
    F::S
    d::Vector{T}
    vk::Vector{T}
end
function LinearAlgebra.mul!(y, H::HessianMarkowitz, x)
    mul!(H.vk, H.F', x)
    mul!(y, H.F, H.vk)
    @. y += H.d*x
    return nothing
end

function generate_portfolio_data(n, k; rseed=1)
    Random.seed!(1)

    # Σ = F*F' + diag(d)
    F = sprandn(n, k, 0.5)
    d = rand(n) * sqrt(k)
    μ = randn(n)
    γ = 1;
    return F, d, μ, γ
end

function build_qp_matrices(F, d, μ, γ)
    n, k = size(F)
    P = γ * blockdiag(spdiagm(d), spdiagm(ones(k)))
    q = vcat(-μ, zeros(k))
    A = [
        F'                  -spdiagm(ones(k));
        spdiagm(ones(n))    spzeros(n, k)
        ones(1, n)          zeros(1, k)
    ]
    l = vcat(zeros(k), zeros(n), 1)
    u = vcat(zeros(k), Inf*ones(n), 1)

    return P, q, A, l, u
end

function run_trial(n::Int; solvers=[:qp, :op, :custom, :cosmo_indirect, :cosmo_direct, :osqp, :qp_full, :mosek])
    GC.gc()
    BLAS.set_num_threads(Sys.CPU_THREADS)
    filename = "$SAVEFILE-$n-may2025.jld2"
    savefile = joinpath(SAVEPATH, filename)

    k = n ÷ 100
    F, d, μ, γ = generate_portfolio_data(n, k)
    P_eq, q_eq, M_eq, l_eq, u_eq = build_qp_matrices(F, d, μ, γ)
    GC.gc()


    # OSQP
    if :osqp ∈ solvers
        GC.gc()
        osqp_model = OSQP.Model()
        OSQP.setup!(
            osqp_model; P=P_eq, q=q_eq, A=M_eq, l=l_eq, u=u_eq, 
            eps_abs=1e-4, eps_rel=1e-4, verbose=false, time_limit=1800,
        )
        result_osqp = OSQP.solve!(osqp_model)
        @info "\tFinished OSQP, time: $(result_osqp.info.solve_time + result_osqp.info.setup_time)"
    else
        result_osqp = nothing
    end


    # COSMO (indirect)
    if :cosmo_indirect ∈ solvers
        GC.gc()
        model_cosmo_indirect = COSMO.Model()
        cs1 = COSMO.Constraint(M_eq, zeros(n+k+1), COSMO.Box(l_eq, u_eq))
        settings = COSMO.Settings(
            kkt_solver=CGIndirectKKTSolver, 
            verbose=false,
            verbose_timing = true,
            eps_abs=1e-4,
            eps_rel=1e-4,
            time_limit=1800,
        )
        assemble!(model_cosmo_indirect, P_eq, q_eq, cs1, settings=settings)
        result_cosmo_indirect = COSMO.optimize!(model_cosmo_indirect)
        @info "\tFinished COSMO (indirect), time: $(result_cosmo_indirect.times.iter_time + result_cosmo_indirect.times.factor_update_time)"
    else
        result_cosmo_indirect = nothing
    end


    # COSMO
    if :cosmo_direct ∈ solvers
        GC.gc()
        model_cosmo_direct = COSMO.Model()
        cs1 = COSMO.Constraint(M_eq, zeros(n+k+1), COSMO.Box(l_eq, u_eq))
        settings = COSMO.Settings(
            kkt_solver=QdldlKKTSolver,
            verbose=false,
            verbose_timing = true,
            eps_abs=1e-4,
            eps_rel=1e-4,
            time_limit=1800,
        )
        assemble!(model_cosmo_direct, P_eq, q_eq, cs1, settings=settings)
        result_cosmo_direct = COSMO.optimize!(model_cosmo_direct)
        @info "\tFinished COSMO (direct), time: $(result_cosmo_direct.times.iter_time + result_cosmo_direct.times.factor_update_time)"
    else
        result_cosmo_direct = nothing
    end

    # GeNIOS
    options = GeNIOS.SolverOptions(
        verbose=false,
        eps_abs=1e-4,
        eps_rel=1e-4,
        norm_type=Inf,
        sketch_update_iter=10_000,
        max_time_sec=1800.0,
    )

    if :qp ∈ solvers
        GC.gc()
        solver = GeNIOS.QPSolver(P_eq, q_eq, M_eq, l_eq, u_eq; σ=0.0)
        result_qp = solve!(solver; options=options)
    else
        result_qp = nothing
    end
    @info "\tFinished GeNIOS (eq qp), time: $(result_qp.log.setup_time + result_qp.log.solve_time)"
    
    if :qp_full ∈ solvers
        GC.gc()
        P = γ * (Diagonal(d) + F*F')
        M = vcat(I, ones(1, n))
        q = -μ
        l = vcat(zeros(n), ones(1))
        u = vcat(Inf*ones(n), ones(1))
        solver = GeNIOS.QPSolver(P, q, M, l, u; σ=0.0)
        result_qp_full = solve!(solver; options=options)
        @info "\tFinished GeNIOS (full qp), time: $(result_qp_full.log.setup_time + result_qp_full.log.solve_time)"
    else
        result_qp_full = nothing
    end
    
    Fd = Matrix(F)
    # QP with custom operators
    if :op ∈ solvers
        GC.gc()
        P = FastP(Fd, d, γ, zeros(k))
        M = FastM(n)
        q = -μ
        l = vcat(zeros(n), ones(1))
        u = vcat(Inf*ones(n), ones(1))

        solver = GeNIOS.QPSolver(P, q, M, l, u; check_dims=false, σ=0.0);
        result_op = solve!(solver; options=options)
        @info "\tFinished GeNIOS (custom ops), time: $(result_op.log.setup_time + result_op.log.solve_time)"
    else
        result_op = nothing
    end

    # Generic interface with custom f and g 
    if :custom ∈ solvers
        GC.gc()
        params = (; F=F, d=d, μ=μ, γ=γ, tmp=zeros(k))
        # f(x) = γ/2 xᵀ(FFᵀ + D)x - μᵀx
        function f(x, p)
            F, d, μ, γ, tmp = p.F, p.d, p.μ, p.γ, p.tmp
            mul!(tmp, F', x)
            qf = sum(abs2, tmp)
            qf += sum(i->d[i]*x[i]^2, 1:length(x))

            return γ/2 * qf - dot(μ, x)
        end

        #  ∇f(x) = γ(FFᵀ + D)x - μ
        function grad_f!(g, x, p)
            F, d, μ, γ, tmp = p.F, p.d, p.μ, p.γ, p.tmp
            mul!(tmp, F', x)
            mul!(g, F, tmp)
            @. g += d*x
            @. g *= γ
            @. g -= μ
            return nothing
        end

        Hf = HessianMarkowitz(Fd, d, zeros(k))

        # g(z) = I(z)
        function g(z, p)
            T = eltype(z)
            return all(z .>= zero(T)) && abs(sum(z) - one(T)) < 1e-6 ? zero(T) : Inf
        end
        function prox_g!(v, z, ρ, p)
            z_max = maximum(w->abs(w), z)
            l = -z_max - 1
            u = z_max

            # bisection search to find zero of F
            while u - l > 1e-8
                m = (l + u) / 2
                if sum(w->max(w - m, zero(eltype(z))), z) - 1 > 0
                    l = m
                else
                    u = m
                end
            end
            ν = (l + u) / 2
            @. v = max(z - ν, zero(eltype(z)))
            return nothing
        end

        solver = GeNIOS.GenericSolver(
            f, grad_f!, Hf,         # f(x)
            g, prox_g!,             # g(z)
            I, zeros(n);            # M, c: Mx - z = c
            params=params
        )
        result_custom = solve!(solver; options=options)
        @info "\tFinished GeNIOS (generic solver), time: $(result_custom.log.setup_time + result_custom.log.solve_time)"
    else
        result_custom = nothing
    end
    
    # Mosek
    if :mosek ∈ solvers
        GC.gc()
        model = Model(Mosek.Optimizer)
        @variable(model, x[1:n])
        @variable(model, y[1:k])
        @objective(model, Min, sum(i-> γ/2*x[i]^2*d[i] - μ[i]*x[i], 1:n) +  γ/2*y'*y)
        @constraint(model, F'*x .== y)
        @constraint(model, sum(x) == 1)
        @constraint(model, x .≥ 0)
        set_silent(model)
        set_time_limit_sec(model, 1800.0)
        optimize!(model)
        result_mosek = solution_summary(model)
        @info "\tFinished Mosek, time: $(result_mosek.solve_time)"
    else
        result_mosek = nothing
    end
    
    # save data 
    save(savefile,
        "result_qp", result_qp,
        "result_qp_full", result_qp_full,
        "result_op", result_op,
        "result_custom", result_custom,
        "result_cosmo_indirect", result_cosmo_indirect,
        "result_cosmo_direct", result_cosmo_direct,
        "result_osqp", result_osqp,
        "result_mosek", result_mosek,
    )

    GC.gc()
    name_logs = [
        (:qp, result_qp),
        (:qp_full, result_qp_full),
        (:op, result_op),
        (:custom, result_custom),
        (:cosmo_indirect, result_cosmo_indirect),
        (:cosmo_direct, result_cosmo_direct),
        (:osqp, result_osqp),
        (:mosek, result_mosek),
    ]
    to_remove = Set{Symbol}()
    for (name, log) in name_logs
        if !(
            (name ∈ [:qp, :qp_full, :op, :custom] && !isnothing(log) && log.status == :OPTIMAL) || 
            (name ∈ [:cosmo_indirect, :cosmo_direct] && !isnothing(log) && log.status == :Solved) ||
            (name == :osqp && !isnothing(log) && log.info.status == :Solved) ||
            (name == :mosek && !isnothing(log) && log.termination_status == OPTIMAL)
        )
            @info "\t$name timed out"
            push!(to_remove, name)
        end
    end
    return to_remove
end

ns = [250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_024_000]
if !RAN_TRIALS
    run_trial(100)
    solvers = Set([:qp, :op, :custom, :cosmo_indirect, :cosmo_direct, :osqp, :qp_full, :mosek])

    @info "Starting trials..."
    for n in ns
        @info "Starting n=$n"
        @info "\t Solvers: $solvers"
        to_remove = run_trial(n; solvers=solvers)
        n > 16_000 && push!(to_remove, :qp_full)
        setdiff!(solvers, to_remove)
        @info "Finished with n=$n"
    end
end
@info "Finished with all trials!"



## Load data from save file
function get_logs(n)
    savefile = joinpath(SAVEPATH, "$SAVEFILE-$n-may2025.jld2")
    r_qp, r_qpf, r_op, r_custom, rc_indirect, rc_direct, r_osqp, r_mosek = 
        load(savefile, 
            "result_qp", "result_qp_full", "result_op", "result_custom",
            "result_cosmo_indirect", "result_cosmo_direct",
            "result_osqp", "result_mosek"
        )    
    return [
        (:qp, r_qp),
        (:qp_full, r_qpf),
        (:op, r_op),
        (:custom, r_custom),
        (:cosmo_indirect, rc_indirect),
        (:cosmo_direct, rc_direct),
        (:osqp, r_osqp),
        (:mosek, r_mosek),
    ]
end

function get_timing(logs)
    setup_times = Vector{Float64}(undef, 0)
    solve_times = Vector{Float64}(undef, 0)

    for (name, log) in logs
        if name ∈ [:qp, :qp_full, :op, :custom] && !isnothing(log) && log.status == :OPTIMAL
            push!(setup_times, log.log.setup_time)
            push!(solve_times, log.log.solve_time)
        elseif name ∈ [:cosmo_indirect, :cosmo_direct] && !isnothing(log) && log.status == :Solved
            push!(setup_times, log.times.setup_time)
            push!(solve_times, log.times.iter_time + log.times.factor_update_time)
        elseif name == :osqp && !isnothing(log) && log.info.status == :Solved
            push!(setup_times, log.info.setup_time)
            push!(solve_times, log.info.solve_time)
        elseif name == :mosek && !isnothing(log) && log.termination_status == OPTIMAL
            push!(setup_times, 0.0)
            push!(solve_times, log.solve_time)
        else
            push!(setup_times, NaN)
            push!(solve_times, NaN)
        end
    end

    return setup_times, solve_times
end


timings = zeros(length(ns), 8)
setup_times = zeros(length(ns), 8)
solve_times = zeros(length(ns), 8)
for (i, n) in enumerate(ns)
    logs = get_logs(n)
    setup_time, solve_time = get_timing(logs)
    timings[i,:] .= setup_time .+ solve_time
    setup_times[i,:] .= setup_time
    solve_times[i,:] .= solve_time
end

## Plots! Plots! Plots!
timing_plt = plot(
    ns, 
    timings,
    yaxis=:log, 
    xaxis=:log,
    label=["GeNIOS (eq qp)" "GeNIOS (full qp)" "GeNIOS (cust ops)" "GeNIOS (GenericSolver)"  "COSMO (indirect)" "COSMO (direct)" "OSQP" "Mosek"], 
    xlabel=L"Problem size $n$", 
    ylabel="Total solve time (s)", 
    legend=:bottomright,
    lw=2,
    markershape=[:circle :diamond :utriangle :star5 :circle :cross :cross :square],
    color=[:firebrick :firebrick :coral :coral :mediumblue :mediumblue :black :purple],
    dpi=300,
    yticks=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    minorgrid=true,
)
savefig(timing_plt, joinpath(FIGS_PATH, "5-portfolio-timing-2024.pdf"))

# Table
println("\\begin{tabular}{@{}lrrrrrrrr@{}}")
println("\\toprule")
println("Size & GeNIOS (eq) & GeNIOS (full) & GeNIOS (cust) & GeNIOS (gen) & COSMO (ind) & COSMO (dir) & OSQP & Mosek \\\\")
println("\\midrule")
for (i, n) in enumerate(ns)
    println("$n & $(@sprintf("%.3f", timings[i,1])) & $(@sprintf("%.3f", timings[i,2])) & $(@sprintf("%.3f", timings[i,3])) & $(@sprintf("%.3f", timings[i,4])) & $(@sprintf("%.3f", timings[i,5])) & $(@sprintf("%.3f", timings[i,6])) & $(@sprintf("%.3f", timings[i,7])) & $(@sprintf("%.3f", timings[i,8])) \\\\")
end
