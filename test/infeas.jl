@testset "QP -- infeasible" begin
    # Infeas detection
    options = GeNIOS.SolverOptions(relax=false, max_iters=1_000, verbose=false)


    # Primal infeasible
    n = 200
    P = I
    q = ones(n)

    l = vcat(ones(n), 0.0)
    u = vcat(2*ones(n), 0.0)
    M = vcat(I, ones(n)')
    solver = GeNIOS.QPSolver(P, q, M, l, u)
    res = solve!(solver; options=options)
    @test GeNIOS.primal_infeasible(solver, options)
    @test !GeNIOS.dual_infeasible(solver, options)
    @test res.status == :INFEASIBLE

    # Dual infeasible
    n = 100
    P = sparse(Matrix(I, n, n))
    P[end,end] = 0.0
    # q = vcat(1.0, zeros(n-1))
    q = ones(n)
    M = I
    u = ones(n)
    l = -Inf * ones(n)
    solver = GeNIOS.QPSolver(P, q, M, l, u)
    res = solve!(solver; options=options)

    @test !GeNIOS.primal_infeasible(solver, options)
    @test GeNIOS.dual_infeasible(solver, options)
    @test res.status == :DUAL_INFEASIBLE
end