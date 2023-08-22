# Signal Decomposition
using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, BandedMatrices
using Plots, LaTeXStrings
include("utils.jl")

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

FIGS_PATH = joinpath(@__DIR__, "figures")

## Generating the problem data
Random.seed!(1)
T, K = 500, 3
t = range(1, T)

# component 1: noise
s1 = 0.25 * randn(length(t))

# component 2: sine wave
s2 = @. sin(2π * t * 5/T)

# component 3: square wave
# s3 = @. (t % 80 < 50) .|> x -> x > 0 ? 1.0 : -1.0
s3 = @. 2π  * t * 3/T .|> x -> x % 2π < π ? 2.0 : -2.0

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
params = (; T=T)

# f(x) is a quadratic loss for the noise term
f(x, p) = sum(abs2, x[1:p.T] ) / p.T 

function grad_f!(g, x, p)
    @. g[1:p.T] = 2/p.T * x[1:p.T]
    g[p.T+1:end] .= zero(eltype(x))
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
#   3: in [-1, 1]
function g(z, p)
    @views z1, z2, z3 = z[1:p.T], z[p.T+1:2p.T], z[2p.T+1:end]
    any(.!iszero.(z1)) && return Inf

    gz2 = sum(t->(z2[t+1] - 2z2[t] + z2[t-1])^2, 2:p.T-1)
    
    !all(abs.(z3) .== 2.0) && return Inf
    return gz2
end

# the prox operator for g
function prox_g!(v, z, ρ, p)
        @views z1, z2, z3 = z[1:p.T], z[p.T+1:2p.T], z[2p.T+1:end]
        @views v1, v2, v3 = v[1:p.T], v[p.T+1:2p.T], v[2p.T+1:end]
        θ2 = 1e4

        # Prox for z1
        v[1:p.T] .= zero(eltype(z))

        # Prox for z2
        du = vcat(zeros(1), ones(p.T-2))
        d = vcat(zeros(1), -2*ones(p.T-2), zeros(1))
        dl = vcat(ones(p.T-2), zeros(1))

        A = BandedMatrix(-1 => dl, 0 => d, 1 => du)
        v2 .= (I + θ2/(ρ * p.T) * A'*A) \ z2

        # Prox for z3
        @inline proj(x) = x >= 0.0 ? 2.0 : -2.0
        @. v3 = proj(z3)

    return nothing
end

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
    M, c;                   # M, c: Mx + z = c
    params=params
)
options = GeNIOS.SolverOptions(
    linsys_max_tol=1e-1
)

# Compile pass
solve!(solver, options=GeNIOS.SolverOptions(max_iters=2))                 
# solve
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
decomp_plt = plot(p1, p2, p3, layout=(3,1), size=(1600, 900), dpi=300)
savefig(decomp_plt, joinpath(FIGS_PATH, "6-signal-decomp-components.pdf"))
