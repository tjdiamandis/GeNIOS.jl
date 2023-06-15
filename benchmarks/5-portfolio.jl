using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using Statistics, JLD2
include(joinpath(@__DIR__, "utils.jl"))

Pkg.activate(joinpath(@__DIR__, ".."))
using GeNIOS

SAVEPATH = joinpath(@__DIR__, "saved")
SAVEFILE = joinpath(SAVEPATH, "5-portfolio.jld2")
FIGS_PATH = joinpath(@__DIR__, "figures")


## Generate the problem data
Random.seed!(1)
k = 10
n = 100k
# Σ = F*F' + diag(d)
F = sprandn(n, k, 0.5)
d = rand(n) * sqrt(k)
μ = randn(n)
γ = 1;


## Vanilla QP
P = γ*(F*F' + Diagonal(d))
q = -μ
M = vcat(I, ones(1, n))
l = vcat(zeros(n), ones(1))
u = vcat(Inf*ones(n), ones(1))

# compile
solve!(GeNIOS.QPSolver(P, q, M, l, u); options=GeNIOS.SolverOptions(max_iters=2))

# solve
solver = GeNIOS.QPSolver(P, q, M, l, u)
options = GeNIOS.SolverOptions(
    num_threads=Sys.CPU_THREADS,
)
GC.gc()
result_qp = solve!(solver; options=options)


## QP with custom operators
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
P = FastP(F, d, γ, zeros(k))

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
M = FastM(n)
solver = GeNIOS.QPSolver(P, q, M, l, u);
GC.gc()
result_op = solve!(solver; options=options)


## Generic interface with custom f and g 
# f(x) = γ/2 xᵀ(FFᵀ + D)x - μᵀx
function f(x, F, d, μ, γ, tmp)
    mul!(tmp, F', x)
    qf = sum(w->w^2, tmp)
    qf += sum(i->d[i]*x[i]^2, 1:n)

    return γ/2 * qf - dot(μ, x)
end
f(x) = f(x, F, d, μ, γ, zeros(k))

#  ∇f(x) = γ(FFᵀ + D)x - μ
function grad_f!(g, x, F, d, μ, γ, tmp)
    mul!(tmp, F', x)
    mul!(g, F, tmp)
    @. g += d*x
    @. g *= γ
    @. g -= μ
    return nothing
end
grad_f!(g, x) = grad_f!(g, x, F, d, μ, γ, zeros(k))

# ∇²f(x) = γ(FFᵀ + D)
struct HessianMarkowitz{T, S <: AbstractMatrix{T}} <: HessianOperator
    F::S
    d::Vector{T}
    vk::Vector{T}
end
function LinearAlgebra.mul!(y, H::HessianMarkowitz, x)
    mul!(H.vk, H.F', x)
    mul!(y, H.F, H.vk)
    @. y += d*x
    return nothing
end
Hf = HessianMarkowitz(F, d, zeros(k))

# g(z) = I(z)
function g(z)
    T = eltype(z)
    return all(z .>= zero(T)) && abs(sum(z) - one(T)) < 1e-6 ? 0 : Inf
end
function prox_g!(v, z, ρ)
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

# compile
solve!(GeNIOS.GenericSolver(f, grad_f!, Hf, g, prox_g!, I, zeros(n)); options=GeNIOS.SolverOptions(max_iters=2))

solver = GeNIOS.GenericSolver(
    f, grad_f!, Hf,         # f(x)
    g, prox_g!,             # g(z)
    I, zeros(n)             # M, c: Mx + z = c
)
GC.gc()
result_custom = solve!(solver; options=options)

# Save data
save(SAVEFILE,
    "result_qp", result_qp,
    "result_op", result_op,
    "result_custom", result_custom,
)


## Load data from save file
result_qp, result_op, result_custom = load(SAVEFILE,
    "result_qp", "result_op", "result_custom",
)

results = [result_qp, result_op, result_custom]
any(x.status != :OPTIMAL for x in results) && @warn "Some problems not solved!"

## Plots! Plots! Plots!
log_qp = result_qp.log
log_op = result_op.log
log_custom = result_custom.log

# print timinigs
print_timing("QPSolver", log_qp)
print_timing("QPSolver, custom operators", log_op)
print_timing("GenericSolver", log_custom)

print_timing_table(
    ["QPSolver", "QPSolver, custom operators", "GenericSolver"], 
    [log_qp, log_op, log_custom]
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
add_to_plot!(rp_iter_plot, log_qp.iter_time, log_qp.rp, "QPSolver", :coral);
add_to_plot!(rp_iter_plot, log_op.iter_time, log_op.rp, "QPSolver, custom ops", :purple);
add_to_plot!(rp_iter_plot, log_custom.iter_time, log_custom.rp, "GenericSolver", :red);
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
add_to_plot!(rd_iter_plot, log_qp.iter_time, log_qp.rd, "QPSolver", :coral);
add_to_plot!(rd_iter_plot, log_op.iter_time, log_op.rd, "QPSolver, custom ops", :purple);
add_to_plot!(rd_iter_plot, log_custom.iter_time, log_custom.rd, "GenericSolver", :red);
rd_iter_plot

# pstar = log_high_precision.obj_val[end]
# obj_val_iter_plot = plot(; 
#     dpi=300,
#     legendfontsize=10,
#     labelfontsize=14,
#     yaxis=:log,
#     ylabel=L"$(p-p^\star)/p^\star$",
#     xlabel="Time (s)",
#     legend=:topright,
# )
# add_to_plot!(obj_val_iter_plot, log_pc.iter_time[2:end], (log_pc.obj_val[2:end] .- pstar) ./ pstar, "GeNIOS", :coral);
# add_to_plot!(obj_val_iter_plot, log_npc.iter_time[2:end], (log_npc.obj_val[2:end] .- pstar) ./ pstar, "No PC", :purple);
# add_to_plot!(obj_val_iter_plot, log_exact.iter_time[2:end], (log_exact.obj_val[2:end] .- pstar) ./ pstar, "ADMM (pc)", :red);
# add_to_plot!(obj_val_iter_plot, log_exact_npc.iter_time[2:end], (log_exact_npc.obj_val[2:end] .- pstar) ./ pstar, "ADMM (no pc)", :mediumblue);
# obj_val_iter_plot

savefig(rp_iter_plot, joinpath(FIGS_PATH, "5-portfolio-rp-iter.pdf"))
savefig(rd_iter_plot, joinpath(FIGS_PATH, "5-portfolio-rd-iter.pdf"))
# savefig(obj_val_iter_plot, joinpath(FIGS_PATH, "5-portfolio-obj-val-iter.pdf"))
