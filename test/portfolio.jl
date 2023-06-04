
# Generate Fake Data
Random.seed!(1)
k = 5
n = 100k
F = sprandn(n, k, 0.5)
d = rand(n) * sqrt(k)
μ = randn(n)
γ = 1.0

P = γ*(F*F' + Diagonal(d))
q = -μ

@testset "Portfolio - Generic" begin
    struct HessianMarkowitz{T, S <: AbstractMatrix{T}} <: HessianOperator
        F::S
        d::Vector{T}
        vk::Vector{T}
    end

    function LinearAlgebra.mul!(y, H::HessianMarkowitz, x)
        # y = (FFᵀ + D)x
        mul!(H.vk, H.F', x)
        mul!(y, H.F, H.vk)
        @. y += d*x
        return nothing
    end

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

    Hf = HessianMarkowitz(F, d, zeros(k))

    function g(z)
        T = eltype(z)
        return all(z .>= zero(T)) && abs(sum(z) - one(T)) < 1e-6 ? 0 : Inf
    end

    function prox_g!(v, z, ρ)
        z_max = maximum(w->abs(w), z)
        l = -z_max - 1
        u = z_max

        @inline ff(ν, z) = sum(w->max(w - ν, zero(eltype(z))), z) - 1
        # bisection search to find zero of F
        while u - l > 1e-8
            m = (l + u) / 2
            if ff(m, z) > 0
                l = m
            else
                u = m
            end
        end
        ν = (l + u) / 2

        @. v = max(z - ν, zero(eltype((z))))
        return nothing
    end

    solver = GeNIOS.GenericSolver(
        f, grad_f!, Hf,         # f(x)
        g, prox_g!,             # g(z)
        I, zeros(n);           # A, c: Ax + z = c
        ρ=1.0, α=1.0
    )
    res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, max_iters=1000, eps_abs=1e-6, eps_rel=1e-6, verbose=false))

    # Check optimality
    xk, zk = solver.xk, solver.zk
    @test all(zk .>= 0)                             # pfeas
    @test abs(sum(zk) - 1) < 1e-3                   # pfeas
    @test norm(solver.xk - solver.zk) < 1e-3        # pfeas

    # Checks with zk but could use xk instead
    ind = findall(x->x > 1e-4, zk)
    v = (P*zk)[ind] - μ[ind]
    @test abs(maximum(v) - minimum(v)) < 1e-3     # L(x, ν) dissapears at xstar

end

@testset "Portfolio -- Conic" begin
    K = GeNIOS.IntervalCone(zeros(n+1), vcat(Inf*ones(n), zeros(1)))
    M = vcat(I, ones(1, n))
    c = vcat(zeros(n), ones(1))
    solver = GeNIOS.ConicSolver(
        P, q, K, M, c
    )
    res = solve!(solver; options=GeNIOS.SolverOptions(relax=false, max_iters=1000, eps_abs=1e-6, eps_rel=1e-6, verbose=false))

    # Check optimality
    xk, zk = solver.xk, solver.zk
    zk = zk[1:n]
    @test all(zk .>= 0)                             # pfeas
    @test abs(sum(zk) - 1) < 1e-3                   # pfeas
    @test norm(M*solver.xk - solver.zk - c) < 1e-3        # pfeas

    # Checks with zk but could use xk instead
    ind = findall(x->x > 1e-4, zk)
    v = (P*zk)[ind] - μ[ind]
    @test abs(maximum(v) - minimum(v)) < 1e-3     # L(x, ν) dissapears at xstar
end