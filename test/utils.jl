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

    # ∇f(x) + Aᵀu = 0
    @test all(abs.(v - solver.uk) .<= tol)

    # ∂g(z) + u = 0
    @test all(abs.(@. γ + solver.uk[pos_inds]) .<= tol)
    @test all(abs.(@. -γ + solver.uk[neg_inds]) .<= tol)
    @test all(abs.(solver.uk[zero_inds]) .<= γ)    
    
    return nothing
end