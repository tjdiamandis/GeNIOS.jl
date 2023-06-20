# Signal Decomposition
using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays
using Plots, LaTeXStrings

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

include("utils.jl")

FIGS_PATH = joinpath(@__DIR__, "figures")

## Generating the problem data
Random.seed!(1)
T, K = 500, 3
t = range(1, T)

# component 1: noise
s1 = 0.2 * randn(length(t))

# component 2: quadratic
s2 = @. (3t/T)^2

# component 3: sine wave
s3 = @. sin(2π * t * 5/T)

# observed signal
X = hcat(s1, s2, s3)
y = sum(X, dims=2) |> vec

sig_plt = plot(
    t, 
    y,
    lw=1,
    label="observed signal",
    xlabel="time",
    ylabel="signal",
    legend=:bottomright,
    dpi=300,
    marker=:circle,
    ls=:dash,
    color=:royalblue
)
plot!(sig_plt, 
    t, 
    sum(X[:, 2:end], dims=2) |> vec, 
    label="true signal",
    lw=3,
    color=:royalblue
)
savefig(sig_plt, joinpath(FIGS_PATH, "6-signal-decomp-observed.pdf"))

## Defining the probelem
# We will separate the observed signal $y$ into a sum of $K=3$ signals $x_k$.
# f(x) is a quadratic loss for the noise term
f(x, T) = sum(w->w^2, x[1:T] ) / T 
f(x) = f(x, T)

function grad_f!(g, x, T)
    @. g[1:T] = 2/T * x[1:T]
    g[T+1:end] .= zero(eltype(x))
    return nothing
end
grad_f!(g, x) = grad_f!(g, x, T)

# HessianOperator is block diagonal, with a T x T identity block & two zero blocks
struct HessianSignal <: HessianOperator end
function LinearAlgebra.mul!(y, ::HessianSignal, x)
    T = length(x) ÷ 3
    @. y[1:T] = 2/T * x[1:T]
    y[T+1:end] .= zero(eltype(x))
    return nothing
end
update!(::HessianSignal, ::Solver) = nothing
Hf = HessianSignal()

# Prox for g is separable across components
#   1: forces equality.
#   2: second-order smooth (smooth derivative).
#   3: first difference sparse (piecewise constant).
function g(z, T)
    @views z1, z2, z3 = z[1:T], z[T+1:2T], z[2T+1:end]
    any(.!iszero.(z1)) && return Inf

    gz2 = sum(t->(z2[t+1] - 2z2[t] + z2[t-1])^2, 2:T-1)
    gz3 = sum(t->(z3[t+1] - z3[t])^2, 1:T-1)
    return gz2 + gz3
end
g(z) = g(z, T)

# Matrix for g² 
A = spdiagm(
    -1 => vcat(ones(T-2), zeros(1)), 
    0 => vcat(-2*ones(T-1), zeros(1)), 
    1 => vcat(zeros(1), ones(T-2))
)
M2 = Matrix(A'*A)

# Matrix for g³
A = vcat(
    spdiagm(
        0 => -1*ones(T),
        100 => ones(T-100), 
        -100 => vcat(zeros(T-200), ones(100))
    ), 
    spdiagm(
        0 => -1*ones(T-1),
        1 => ones(T-1)
    )
)
M3 = Matrix(A'*A)

# the prox operator for g, using ProximalOperators.jl
function prox_g!(v, z, ρ, T; M2, M3)
    Threads.@threads for k in 1:3
        if k == 1
            ## Prox for z1
            v[1:T] .= zero(eltype(z))
        elseif k == 2
            ## Prox for z2 (second order smooth)
            ## g²(z²) = θ₂/T * ||Az²||²
            z2 = @view z[T+1:2T]
            v2 = @view v[T+1:2T]
            θ2 = 1e6

            v2 .=  (I + θ2/(ρ * T) * M2) \ z2
        elseif k == 3
            ## Prox for z3 (periodic)
            ## g³(z³) = θ₃/T * ||Az³||²
            z3 = @view z[2T+1:end]
            v3 = @view v[2T+1:end]
            θ3 = 5e1

            v3 .= (I + θ3/(ρ * T) * M3) \ z3
        end
    end

    return nothing
end
prox_g!(v, z, ρ) = prox_g!(v, z, ρ, T; M2=M2, M3=M3)

# The constraints
# NOTE: M is a highly structured, but we do not exploit this fact here
IT = sparse(Matrix(1.0I, T, T))
_0 = spzeros(T, T)
M = [   
        IT  IT  IT; 
        _0  IT  _0; 
        _0  _0  IT
    ]
c = vcat(y, zeros(T), zeros(T));

## Solving the problem
solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    g, prox_g!,             # g(z)
    M, c                    # M, c: Mx + z = c
)
options = GeNIOS.SolverOptions(
    eps_abs=1e-6, 
    print_iter=100,
    num_threads=1,
    max_iters=10_000
)
# Compile pass
solve!(solver, options=GeNIOS.SolverOptions(max_iters=2))                 
# Real solve
result = solve!(solver, options=options)

x1 = solver.xk[1:T]
x2 = solver.xk[T+1:2T]
x3 = solver.xk[2T+1:end];


## Plotting the results
res_plt = plot(
    t, 
    y,
    lw=1,
    label="observed signal",
    xlabel="time",
    ylabel="signal",
    legend=:bottomright,
    dpi=300,
    marker=:circle,
    ls=:dash
);
plot!(res_plt, 
    t, 
    sum(X[:, 2:end], dims=2) |> vec, 
    label="true signal",
    lw=3,
    color=:blue
);
plot!(res_plt, 
    t, 
    x2 + x3, 
    label="reconstructed signal",
    lw=3,
    color=:coral1
)
savefig(res_plt, joinpath(FIGS_PATH, "6-signal-decomp-reconstructed.pdf"))


# Visualizing results each component
p1 = plot(
    t, 
    X[:, 1],
    lw=1,
    label="x1 true",
    xlabel="time",
    ylabel="signal",
    legend=:bottomright,
    dpi=300,
    marker=:circle,
    ls=:dash,
    color=:royalblue
);
plot!(p1, 
    t, 
    x1, 
    label="x1 estimated",
    lw=3,
    color=:coral1
);
p2 = plot(
    t, 
    X[:, 2],
    lw=3,
    label="x2 true",
    xlabel="time",
    ylabel="signal",
    legend=:bottomright,
    dpi=300,
    ls=:dash,
    color=:royalblue
);
plot!(p2, 
    t, 
    x2, 
    label="x2 estimated",
    lw=3,
    color=:coral1
);
p3 = plot(
    t, 
    X[:, 3],
    lw=3,
    label="x3 true",
    xlabel="time",
    ylabel="signal",
    legend=:bottomright,
    dpi=300,
    ls=:dash,
    color=:royalblue
);
plot!(p3, 
    t, 
    x3, 
    label="x3 estimated",
    lw=3,
    color=:coral1
);
decomp_plt = plot(p1, p2, p3, layout=(3,1))
savefig(decomp_plt, joinpath(FIGS_PATH, "6-signal-decomp-components.pdf"))


# the prox operator for g, using ProximalOperators.jl
function prox_g_single!(v, z, ρ, T)
    ## Prox for z1
    v[1:T] .= zero(eltype(z))

    ## Prox for z2 (second order smooth)
    ## g²(z²) = θ₂/T * ||Az²||²
    z2 = @view z[T+1:2T]
    v2 = @view v[T+1:2T]
    θ2 = 1e6

    v2 .=  (I + θ2/(ρ * T) * M2) \ z2

    ## Prox for z3 (periodic)
    ## g³(z³) = θ₃/T * ||Az³||²
    z3 = @view z[2T+1:end]
    v3 = @view v[2T+1:end]
    θ3 = 5e1

    v3 .= (I + θ3/(ρ * T) * M3) \ z3

    return nothing
end
prox_g_single!(v, z, ρ) = prox_g_single!(v, z, ρ, T)
solver_single = GeNIOS.GenericSolver(
    f, grad_f!, Hf,             # f(x)
    g, prox_g_single!,          # g(z)
    M, c                        # M, c: Mx + z = c
)
# options_single = GeNIOS.SolverOptions(eps_abs=1e-5, print_iter=100, num_threads=1)
result_single = solve!(solver_single, options=options)


## Timing plots
log_threads = result.log
log_single = result_single.log

print_timing_table(
    ["Parallel Prox", "Standard"], 
    [log_threads, log_single]
)

rp_iter_plot = plot(; 
    dpi=300,
    legendfontsize=10,
    labelfontsize=14,
    yaxis=:log,
    ylabel=L"Primal Residual $\ell_2$ Norm",
    xlabel="Time (s)",
    legend=:topright,
)
add_to_plot!(rp_iter_plot, log_threads.iter_time, log_threads.rp, "Multithreaded prox", :coral);
add_to_plot!(rp_iter_plot, log_single.iter_time, log_single.rp, "Single-threaded prox", :purple);
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
add_to_plot!(rd_iter_plot, log_threads.iter_time, log_threads.rd, "Multithreaded prox", :coral);
add_to_plot!(rd_iter_plot, log_single.iter_time, log_single.rd, "Single-threaded prox", :purple);
rd_iter_plot

savefig(rp_iter_plot, joinpath(FIGS_PATH, "6-signal-decomp-rp.pdf"))
savefig(rd_iter_plot, joinpath(FIGS_PATH, "6-signal-decomp-rd.pdf"))