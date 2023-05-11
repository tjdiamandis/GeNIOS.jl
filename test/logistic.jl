@testset "Logistic - MLSolver" begin
    Random.seed!(1)
    N, n = 2_000, 4_000
    Ã = sprandn(N, n, 0.2)
    @views [normalize!(Ã[:, i]) for i in 1:n-1]
    Ã[:,n] .= 1.0

    xstar = zeros(n)
    inds = randperm(n)[1:100]
    xstar[inds] .= randn(length(inds))
    b̃ = sign.(Ã*xstar + 1e-1 * randn(N))
    b = zeros(N)
    A = Diagonal(b̃) * Ã

    γmax = norm(0.5*A'*ones(N), Inf)
    γ = 0.05*γmax

    # Logistic problem: min ∑ log(1 + exp(aᵢᵀx)) + γ||x||₁
    f2(x) = GeNIOS.log1pexp(x)
    df2(x) = GeNIOS.logistic(x)
    d2f2(x) = GeNIOS.logistic(x) / GeNIOS.log1pexp(x)
    f2conj(x::T) where {T} = (one(T) - x) * log(one(T) - x) + x * log(x)
    λ1 = γ
    λ2 = 0.0
    solver = GeNIOS.MLSolver(f2, df2, d2f2, λ1, λ2, A, b; fconj=f2conj)
    res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, dual_gap_tol=1e-4, verbose=false))
    
    tol = 5e-3
    test_optimality_conditions(solver, tol)

    res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=false, eps_abs=1e-5, eps_rel=1e-5, verbose=false))
    test_optimality_conditions(solver, tol)
end