
@testset "QP -- ConicSolver" begin
    n = 100
    P, q, M, l, u = generate_random_qp(n; rseed=1)
    K = GeNIOS.IntervalCone(l, u)
    c = zeros(length(l))
    solver = GeNIOS.ConicSolver(
        P, q, K, -M, c
    )
    res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, max_iters=1000, eps_abs=1e-6, eps_rel=1e-6, verbose=false))
    xk, zk = solver.xk, solver.zk
    
    tol = 1e-3
    # Check feasibility
    @test norm(-M*xk + zk - c, Inf) <= tol
    @test all(l .<= zk) && all(zk .<= u)
    
    # Check optimality
    fstar = solver.obj_val
    fstar_mosek = -1.443337417081063
    @test abs(fstar - fstar_mosek) <= tol
end