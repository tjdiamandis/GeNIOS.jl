# Signal Decomposition
using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Plots
using ProximalOperators: TotalVariation1D, prox!
using BandedMatrices

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

## Generating the problem data
Random.seed!(1)
T, K = 600, 3
t = range(1, T)

# component 1: noise
s1 = 0.2 * randn(T)

# component 2: sine wave
s2 = @. sin(2π * t * 5/T)

# component 3: square wave
s3 = @. 2π  * t * 4/T .|> x -> x % 2π < π ? 1 : -1
# # component 3: quadratic
# s3 = @. (3t/T)^2

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
    legend=:topright,
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

    gz2 = sum(t->(z2[t+1] - 2z[t] + z[t-1])^2, 2:T-1)
    gz3 = sum(abs, diff(z3)) / (T-1)
    return gz2 + gz3
end
g(z) = g(z, T)

# the prox operator for g, using ProximalOperators.jl
function prox_g!(v, z, ρ, T)
    @views z1, z2, z3 = z[1:T], z[T+1:2T], z[2T+1:end]
    @views v1, v2, v3 = v[1:T], v[T+1:2T], v[2T+1:end]
    θ2, θ3 = 5e3, 5

    Threads.@threads for k in 1:3
        if k == 1
            ## Prox for z1
            v1 .= zero(eltype(z))
        elseif k == 2
            ## Prox for z2
            ## g²(z²) = γ₂/T * ||Az||²
            du = vcat(zeros(1), ones(T-2))
            d = vcat(zeros(1), -2*ones(T-2), zeros(1))
            dl = vcat(ones(T-2), zeros(1))

            ## Use banded matrix for O(T) solve time
            A = BandedMatrix(-1 => dl, 0 => d, 1 => du)
            F = cholesky(I + θ2/(ρ * T) * A'*A)
            ldiv!(v2, F, z2)
        else
            ## Prox for z3
            ϕ³ = TotalVariation1D(θ3/T)
            prox!(v3, ϕ³, z3, 1/ρ)
        end
    end

    return nothing
end
prox_g!(v, z, ρ) = prox_g!(v, z, ρ, T)

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
    max_iters=2_000
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
    legend=:topright,
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


# Visualizing results each component
p1 = plot(
    t, 
    X[:, 1],
    lw=1,
    label="x1 true",
    xlabel="time",
    ylabel="signal",
    legend=:topright,
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
    legend=:topright,
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
    legend=:topright,
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


# the prox operator for g, using ProximalOperators.jl
function prox_g_single!(v, z, ρ, T)
    @views z1, z2, z3 = z[1:T], z[T+1:2T], z[2T+1:end]
    @views v1, v2, v3 = v[1:T], v[T+1:2T], v[2T+1:end]
    θ2, θ3 = 5e3, 5

    ## Prox for z1
    v1 .= zero(eltype(z))

    ## Prox for z2
    ## g²(z²) = γ₂/T * ||Az||²
    du = vcat(zeros(1), ones(T-2))
    d = vcat(zeros(1), -2*ones(T-2), zeros(1))
    dl = vcat(ones(T-2), zeros(1))

    ## Use banded matrix for O(T) solve time
    A = BandedMatrix(-1 => dl, 0 => d, 1 => du)
    F = cholesky(I + θ2/(ρ * T) * A'*A)
    ldiv!(v2, F, z2)

    ## Prox for z3
    ϕ³ = TotalVariation1D(θ3/T)
    prox!(v3, ϕ³, z3, 1/ρ)

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

