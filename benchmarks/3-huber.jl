using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using JLD2
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

const SAVEPATH = joinpath(@__DIR__, "saved", "3-huber")
const FIGS_PATH = joinpath(@__DIR__, "figures")

const RAN_TRIALS = true

## Generating the problem data
function generate_data_huber(; 
    N, n, rseed=1, outliers_frac=0.05, outliers_factor=10, sparsity_frac=0.1, 
    noise_factor=0.1
)
    Random.seed!(rseed)
    BLAS.set_num_threads(Sys.CPU_THREADS)
    A = randn(N, n)
    A .-= sum(A, dims=1) ./ N
    normalize!.(eachcol(A))
    xstar = sprandn(n, sparsity_frac)
    b = A*xstar + noise_factor*randn(N)

    # Add outliers
    outlier_inds = randperm(N)[1:round(Int, outliers_frac*N)]
    b[outlier_inds] += outliers_factor*rand((-1,1), length(outlier_inds))
    return A, b, xstar
end

function create_qp_data(A, b, λ1)
    N, n = size(A)
    IN = sparse(I, N, N)
    In = sparse(I, n, n)
    P = blockdiag(spzeros(n,n), 2IN, spzeros(2N+n,2N+n))
    q = vcat(zeros(n+N), 2*ones(2N), λ1*ones(n))

    Z_Nn = spzeros(N,n)
    Z_NN = spzeros(N,N)
    Z_nN = spzeros(n,N)
    Aqp = [A       -IN     -IN    IN    Z_Nn;
        Z_Nn    Z_NN    IN     Z_NN  Z_Nn;
        Z_Nn    Z_NN    Z_NN   IN    Z_Nn;
        In      Z_nN    Z_nN   Z_nN  In;
        -In     Z_nN    Z_nN   Z_nN  In]
    l = vcat(b, zeros(2N +2n))
    u = vcat(b, Inf*ones(2N + 2n))
    return P, q, Aqp, l, u
end

# Custom convergence: QPSolver
function GeNIOS.converged(solver::GeNIOS.ConicSolver, options::GeNIOS.SolverOptions)::Bool
    N = nnz(solver.data.P)
    n = (solver.data.n - 3N) ÷ 2
    @inline abs2_∇fi(v, x, λ1, tol) = abs(x) < tol ? abs2(max(abs(v) - λ1, 0.0)) : abs2(v + λ1*sign(x))
    @inline df(x::Float64) = abs(x) <= 1 ? 2.0*x : 2.0*sign(x)

    A = @view solver.data.M[1:N, 1:n]
    λ1 = solver.data.q[end]
    b = @view solver.data.K.l[1:N]
    x = @view solver.xk[1:n]
    vN = @view solver.cache.vm[1:N]
    # vN::Vector{Float64} = solver.cache.vm
    # vn::Vector{Float64} = solver.cache.vn
    vn = @view solver.cache.vn[1:n]
    
    # vn = Aᵀℓ'(Ax - b)
    # Mxk[1:N] = Ax - r - s + t
    @views vN .= solver.Mxk[1:N] .+ solver.xk[n+1:n+N] .+ solver.xk[n+N+1:n+2N] .- solver.xk[n+2N+1:n+3N] .- b
    @views vN .= df.(vN)
    @views mul!(vn, A', vN)

    @views vn .= abs2_∇fi.(vn, x, λ1, options.eps_abs)
    return norm(vn) <= options.eps_abs
end

# Custom convergence: MLSolver
function GeNIOS.converged(solver::GeNIOS.MLSolver, options::GeNIOS.SolverOptions)
    @inline abs2_∇fi(v, x, λ1, tol) = abs(x) < tol ? abs2(max(abs(v) - λ1, 0.0)) : abs2(v + λ1*sign(x))

    vn::Vector{Float64} = solver.cache.vn
    vN::Vector{Float64} = solver.cache.vN

    # pred = Ax - b
    vN .= solver.pred
    vN .= solver.data.df.(vN)
    mul!(vn, solver.data.Adata', vN)

    vn .= abs2_∇fi.(vn, solver.zk, solver.λ1, options.eps_abs)
    return norm(vn) <= options.eps_abs
end

function run_trial(n)
    N = n ÷ 2
    A, b, xstar = generate_data_huber(N=N, n=n)

    # Reguarlization parameters
    λ1_max = norm(A'*b, Inf)
    λ1 = 0.1*λ1_max
    λ2 = 0.0


    # Solve with MLSolver
    GC.gc()
    # ML Solver functions
    f(x) = abs(x) <= 1 ? x^2 : 2.0*abs(x) - 1.0
    df(x) = abs(x) <= 1 ? 2x : 2.0*sign(x)
    d2f(x) = abs(x) <= 1 ? 2.0 : 0.0

    # compile
    solve!(GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b); options=GeNIOS.SolverOptions(max_iters=2, verbose=false))
    solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        verbose= n > 5_000 ? true : false,
        relax=true,
        α=1.6,
        precondition=false,
        eps_abs=1e-4,
        ρ0=10.0,
    )
    GC.gc()
    result_ml = solve!(solver; options=options)
    result_ml.status != :OPTIMAL && @warn "n: $n: MLSolver did not converge!"


    # Solve with QPSolver
    GC.gc()
    P, q, Aqp, l, u = create_qp_data(A, b, λ1)

    # compile
    solve!(GeNIOS.QPSolver(P, q, Aqp, l, u); options=GeNIOS.SolverOptions(max_iters=2, verbose=false))
    solver_qp = GeNIOS.QPSolver(P, q, Aqp, l, u)
    options_qp = GeNIOS.SolverOptions(
        verbose=false,
        relax=true,
        α=1.6,
        precondition=false,
        eps_abs=1e-4,
    )
    GC.gc()
    result_qp = solve!(solver_qp; options=options_qp)
    result_qp.status != :OPTIMAL && @warn "n: $n: QPSolver did not converge!"

    savefile = joinpath(SAVEPATH, "huber-$n.jld2")
    save(savefile, 
        "result_ml", result_ml,
        "result_qp", result_qp
    )
    
    GC.gc()
    return solver, solver_qp
end

ns = [250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
if !RAN_TRIALS
    run_trial(100)
    for n in ns
        run_trial(n)
        @info "Finished with n=$n"
    end
end

function get_logs(n)
    savefile = joinpath(SAVEPATH, "huber-$n.jld2")
    r_ml, r_qp = load(savefile, "result_ml", "result_qp")
    
    if r_ml.status != :OPTIMAL || r_qp.status != :OPTIMAL
        @warn "n = $n: Some problems not solved!"
    end
    
    return r_ml, r_qp
end

function get_timing(r_ml, r_qp)
    log_ml = r_ml.log
    log_qp = r_qp.log
    
    setup_times = [
        log_ml.setup_time, 
        log_qp.setup_time,
    ]

    solve_times = [
        log_ml.solve_time, 
        log_qp.solve_time,
    ]

    linsys_times = [
        sum(log_ml.linsys_time), 
        sum(log_qp.linsys_time),
    ]

    return setup_times, solve_times, linsys_times
end

setup_times = zeros(length(ns), 2)
solve_times = zeros(length(ns), 2)
linsys_times = zeros(length(ns), 2)
for (i, n) in enumerate(ns)
    logs = get_logs(n)
    setup_time, solve_time, linsys_time = get_timing(logs...)

    setup_times[i,:] .= setup_time
    solve_times[i,:] .= solve_time
    linsys_times[i,:] .= linsys_time
end

timing_plt = plot(
    ns, 
    [linsys_times, solve_times], 
    yaxis=:log, 
    xaxis=:log,
    label=["MLSolver linsys" "QPSolver linsys" "MLSolver solve" "QPSolver solve"], 
    xlabel=L"Problem size $n$",
    ylabel="Time (s)", 
    legend=:bottomright,
    lw=2,
    markershape=[:utriangle :square :dtriangle :circle],
    dpi=300,
    yticks=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    xticks=[10^2.5, 1000, 10^3.5, 10000],
    color=[:coral :indigo :coral :indigo],
)
savefig(timing_plt, joinpath(FIGS_PATH, "3-huber-timing.pdf"))


n = 16_000
result_ml, result_qp = get_logs(n)
log_ml = result_ml.log
log_qp = result_qp.log

# Print timings
print_timing("MLSolver", log_ml)
print_timing("QPSolver", log_qp)

names = ["MLSolver", "QPSolver"]
logs = [log_ml, log_qp]
print_timing_table(names, logs)

time_ml = cumsum(log_ml.linsys_time)
time_qp = cumsum(log_qp.linsys_time)

# Plot things
# TODO: add custom convergence criterion so looking at the same thing
resid_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Residual $\ell_2$ Norm",
    xlabel="Time (s)",
    legend=:topright,
)
add_to_plot!(resid_iter_plot, time_ml, log_ml.rp, "MLSolver primal", :coral);
add_to_plot!(resid_iter_plot, time_qp, log_qp.rp, "QPSolver primal", :purple);
add_to_plot!(resid_iter_plot, time_ml, log_ml.rd, "MLSolver dual", :coral; style=:dash);
add_to_plot!(resid_iter_plot, time_qp, log_qp.rd, "QPSolver dual", :purple; style=:dash);
resid_iter_plot

savefig(resid_iter_plot, joinpath(FIGS_PATH, "3-huber-residuals.pdf"))
