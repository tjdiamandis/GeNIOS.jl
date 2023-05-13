#=
# Signal Decomposition

=#
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", "docs"))
using GeNIOS
using Random, LinearAlgebra, SparseArrays
using Plots
using ProximalOperators: TotalVariation1D, prox!
using BandedMatrices

#=
## Generating the problem data
=#
Random.seed!(1)
T, K = 200, 3
t = range(0, 1000, T)

## component 2: sine wave
s2 = @. sin(2π * t * 1/500)

## component 3: square wave
square(x) = x % 2π < π ? 1 : -1
s3 = @. square(2π  * t * 1/450)

## component 1: noise
s1 = 0.1 * randn(T)

## observed signal
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
display(sig_plt)

#=
## Defining the probelem
We will separate the observed signal $y$ into a sum of $K=3$ signals $x_k$.
The first is mean-squared small (noise). 
The second is second-order smooth (smooth derivative).
The third is first difference sparse (piecewise constant).
This problem can be phrased as

$$
\begin{array}{ll}
\text{minimize}  & (1/T)\|y - x^2 - x^3\|_2^2 + \gamma_2 \phi_2(x^2) + \gamma_3 \phi_1(x^3),
\end{array}
$$
where 
$$
\phi_2(x) = \frac{1}{T-2}\sum_{i=2}^{T-1} (x_{t+1} - 2x_t + x_{t-1})^2
$$ 
and 
$$
\phi_3(x) = \frac{1}{T-1}\sum_{i=1}^{T-1} \lvert x_{t+1} - x_t \rvert
$$.

We will solve this problem using the genertic solver interface, with the first
term as $f(x)$ and the second and third terms as $g(z)$.
Specifically, this problem can be rephrased as

$$
\begin{array}{ll}
\text{minimize}  & (1/T)\|x^1\|_2^2 + I_{\{0\}}(z^1) + \gamma_2 \phi_2(z^2) + \gamma_3 \phi_1(z^3),
\text{subject to} & x^1 + x^2 + x^3 - z^1 = y \\
&& x^2 - z^2 = 0 \\
&& x^3 - z^3 = 0,
\end{array}
$$

where $I_{\{0\}}(z^1)$ is the indicator function for the zeros vector.
We note that the proximal operators for the second, third, and fourth terms can 
be parallelized.

First we define $f(x)$, which is just a quadratic.
=#
f(x, T) = sum(w->w^2, x[1:T] ) / T 
f(x) = f(x, T)

function grad_f!(g, x, T)
    @. g[1:T] = 2/T * x[1:T]
    g[T+1:end] .= zero(eltype(x))
    return nothing
end
grad_f!(g, x) = grad_f!(g, x, T)

#=
The `HessianOperator` here is block diagonal, with a $T \times T$ identity block,
followed by two zero blocks.
=#
struct HessianSignal <: HessianOperator end
function LinearAlgebra.mul!(y, ::HessianSignal, x)
    T = length(x) ÷ 3
    @. y[1:T] = 2/T * x[1:T]
    y[T+1:end] .= zero(eltype(x))
    return nothing
end
update!(::HessianSignal, ::Solver) = nothing
Hf = HessianSignal()

#=
### A custom proximal operator
Now, we define $g$ and its proximal operator. Here, we will take advantage of
parallelization, as this problem is clearly separable across components. Further
performance improvements could be made for these proximal operators, which we
avoid for simplicity.
=#
function g(z, T)
    @views z1, z2, z3 = z[1:T], z[T+1:2T], z[2T+1:end]
    any(.!iszero.(z1)) && return Inf

    gz2 = sum(t->(z2[t+1] - 2z[t] + z[t-1])^2, 2:T-1)
    gz3 = sum(abs, diff(z3)) / (T-1)
    return gz2 + gz3
end
g(z) = g(z, T)

## the prox operator for g, using ProximalOperators.jl
function prox_g!(v, z, ρ, T)
    @views z1, z2, z3 = z[1:T], z[T+1:2T], z[2T+1:end]
    @views v1, v2, v3 = v[1:T], v[T+1:2T], v[2T+1:end]

    Threads.@threads for k in 1:3
        if k == 1
            ## Prox for z1
            v1 .= zero(eltype(z))
        elseif k == 2
            ## Prox for z2
            ## g²(z²) = θ₂/T * ||Az||²
            du = vcat(zeros(1), ones(T-2))
            d = vcat(zeros(1), -2*ones(T-2), zeros(1))
            dl = vcat(ones(T-2), zeros(1))

            ## Use banded matrix for O(T) solve time
            A = BandedMatrix(-1 => dl, 0 => d, 1 => du)
            F = cholesky(I + 1e3/(ρ * T) * A'*A)
            ldiv!(v2, F, z2)
        else
            ## Prox for z3
            ϕ³ = TotalVariation1D(1/T)
            prox!(v3, ϕ³, z3, 1/ρ)
        end
    end

    return nothing
end
prox_g!(v, z, ρ) = prox_g!(v, z, ρ, T)



#=
### The constraints
Note that $M$ is a highly structured matrix. We could use this fact to speed up
the operators $M$ and $M^T$, but we do not in this example.
=#

IT = sparse(Matrix(1.0I, T, T))
_0 = spzeros(T, T)
M = [   
        IT  IT  IT; 
        _0  IT  _0; 
        _0  _0  IT
    ]
c = vcat(y, zeros(T), zeros(T))

#=
### Solving the problem
=#
solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    g, prox_g!,             # g(z)
    M, c;                   # M, c: Mx + z = c
    ρ=1.0, α=1.0
)
res = solve!(solver, options=GeNIOS.SolverOptions(eps_abs=1e-5))

x1 = solver.xk[1:T]
x2 = solver.xk[T+1:2T]
x3 = solver.xk[2T+1:end];


#=
## Plotting the results
=#
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
)
plot!(res_plt, 
    t, 
    sum(X[:, 2:end], dims=2) |> vec, 
    label="true signal",
    lw=3,
    color=:blue
)
plot!(res_plt, 
    t, 
    x2 + x3, 
    label="reconstructed signal",
    lw=3,
    color=:coral1
)
display(res_plt)

#=
### Visualizing each component
=#
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
)
plot!(p1, 
    t, 
    x1, 
    label="x1 estimated",
    lw=3,
    color=:coral1
)

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
)
plot!(p2, 
    t, 
    x2, 
    label="x2 estimated",
    lw=3,
    color=:coral1
)

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
)
plot!(p3, 
    t, 
    x3, 
    label="x3 estimated",
    lw=3,
    color=:coral1
)
decomp_plt = plot(p1, p2, p3, layout=(3,1))
display(decomp_plt)