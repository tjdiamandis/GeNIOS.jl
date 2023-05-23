function test_optimality_conditions(solver::GeNIOS.MLSolver, tol)
    xstar = solver.zk
    v = solver.cache.vn2
    γ = solver.λ1

    # v = Aᵀ(∇f(Ax - b)) + λ₂x
    mul!(solver.cache.vN, solver.data.Adata, xstar)
    solver.cache.vN .-= solver.data.bdata
    @. solver.cache.vN = solver.data.df(solver.cache.vN)
    mul!(v, solver.data.Adata', solver.cache.vN)
    v .+= solver.λ2 .* xstar

    pos_inds = findall(x->x > tol, xstar)
    neg_inds = findall(x->x < -tol, xstar)
    zero_inds = findall(x->abs(x) <= tol, xstar)

    # Test optimality for x
    @test all(abs.(v[zero_inds]) .<= γ + tol)
    @test all(abs.(v[pos_inds] .+ γ) .<= tol)
    @test all(abs.(v[neg_inds] .- γ) .<= tol)

    # Test generic optimality conditions
    xstar = solver.xk

    mul!(solver.cache.vN, solver.data.Adata, xstar)
    solver.cache.vN .-= solver.data.bdata
    @. solver.cache.vN = solver.data.df(solver.cache.vN)
    mul!(v, solver.data.Adata', solver.cache.vN)
    v .+= solver.λ2 .* xstar

    # ∇f(x) + ρ*Aᵀu = 0
    @test all(abs.(v + solver.ρ*solver.uk) .<= tol)

    # ∂g(z) - ρ*u = 0
    @test all(abs.(@. γ - solver.ρ*solver.uk[pos_inds]) .<= tol)
    @test all(abs.(@. -γ - solver.ρ*solver.uk[neg_inds]) .<= tol)
    @test all(abs.(solver.ρ*solver.uk[zero_inds]) .<= γ)    
    
    return nothing
end


function generate_random_qp(n; rseed=1)
    Random.seed!(rseed)
    m = 10n
    α = 1e-2
    P = Diagonal(vcat(α*ones(n), ones(n)))
    q = vcat(randn(n), zeros(n))
    MM = sprandn(n, n, 0.15)
    M = [sprandn(m, n, 0.15) zeros(m, n) ; MM I]
    u = vcat(rand(m), zeros(n))
    l = vcat(-rand(m), zeros(n))

    return P, q, M, l, u
end