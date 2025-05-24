
@testset "QP -- ConicSolver" begin
    n = 100
    P, q, M, l, u = generate_random_qp(n; rseed=1)
    K = GeNIOS.IntervalCone(l, u)
    c = zeros(length(l))
    solver = GeNIOS.ConicSolver(
        P, q, K, M, c
    )
    solver_qp = GeNIOS.QPSolver(P, q, M, l, u)

    # Check QP interface
    @inline function check_equals(a::T, b::T) where T
        isempty(fieldnames(T)) && return a == b
        for fn in fieldnames(T)
            fn == :ref && continue
            x, y = getfield(a, fn), getfield(b, fn)
            check_equals(x, y) || return false
        end
        return true
    end
    @test check_equals(solver, solver_qp)

    res = solve!(solver; options=GeNIOS.SolverOptions(relax=false, max_iters=1000, eps_abs=1e-6, eps_rel=1e-6, verbose=false))
    xk, zk = solver.xk, solver.zk
    
    tol = 1e-3
    # Check feasibility
    @test norm(M*xk - zk - c, Inf) <= tol
    @test all(l .<= zk) && all(zk .<= u)
    
    # Check optimality
    fstar = solver.obj_val
    fstar_mosek = -1.4751119028013415
    @test abs(fstar - fstar_mosek) <= tol
    @test res.status == :OPTIMAL
end