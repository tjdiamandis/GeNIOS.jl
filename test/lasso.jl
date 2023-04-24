
function test_optimality_conditions_lasso(solver, tol, A, b; λ2 = 0.0)
    # z is the sparse solution
    xstar = solver.zk

    pos_inds = findall(x->x > tol, xstar)
    neg_inds = findall(x->x < -tol, xstar)
    zero_inds = findall(x->abs(x) <= tol, xstar)
    v = A'*(A*xstar - b) + (λ2 .* xstar)
    
    # Test optimality for x
    @test all(abs.(v[zero_inds]) .<= γ + tol)
    @test all(abs.(v[pos_inds] .+ γ) .<= tol)
    @test all(abs.(v[neg_inds] .- γ) .<= tol)

    # Test generic optimality conditions
    # ∇f(x) + Aᵀu = 0
    @test all(abs.(A'*(A*solver.xk - b) + λ2*solver.xk - solver.uk) .<= tol)

    # ∂g(z) + u = 0
    @test all(abs.(@. γ + solver.uk[pos_inds]) .<= tol)
    @test all(abs.(@. -γ + solver.uk[neg_inds]) .<= tol)
    @test all(abs.(solver.uk[zero_inds]) .<= γ)    
    
    return nothing
end

function test_optimality_conditions(solver::GeNIOS.MLSolver, tol)
    xstar = solver.zk
    v = solver.cache.vn2

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
    @test all(v - solver.uk .<= tol)

    # ∂g(z) + u = 0
    @test all(abs.(@. γ + solver.uk[pos_inds]) .<= tol)
    @test all(abs.(@. -γ + solver.uk[neg_inds]) .<= tol)
    @test all(abs.(solver.uk[zero_inds]) .<= γ)    
    
    return nothing
end




Random.seed!(1)
n, p = 2_000, 4_000
r = 1000
# A = randn(n, r) * Diagonal(.96 .^ (0:r-1)) * randn(r, p)
A = randn(n, p)
A .-= sum(A, dims=1) ./ n
normalize!.(eachcol(A))
xstar = sprandn(p, 0.1)
b = A*xstar + 1e-3*randn(n)

# Lasso problem: min 1/2 ||Ax - b||² + γ||x||₁
γmax = norm(A'*b, Inf)
γ = 0.05*γmax


@testset "Generic" begin
    # Lasso Hessian operator
    # 1/2||Ax - b|| + γ||x||₁
    # ∇²f(x) = AᵀA

    struct HessianLasso{T, S <: AbstractMatrix{T}} <: HessianOperator
        A::S
        vm::Vector{T}
    end
    function LinearAlgebra.mul!(y, H::HessianLasso, x)
        mul!(H.vm, H.A, x)
        mul!(y, H.A', H.vm)
        return nothing
    end
    
    function update!(::HessianLasso, ::AbstractVector)
        return nothing
    end
    
    
    function f(x, A, b, tmp)
        mul!(tmp, A, x)
        @. tmp -= b
        return 0.5 * sum(w->w^2, tmp)
    end
    f(x) = f(x, A, b, zeros(n))
    function grad_f!(g, x, A, b, tmp)
        mul!(tmp, A, x)
        @. tmp -= b
        mul!(g, A', tmp)
        return nothing
    end
    grad_f!(g, x) = grad_f!(g, x, A, b, zeros(n))
    Hf = HessianLasso(A, zeros(n))
    g(z, γ) = γ*sum(x->abs(x), z)
    g(z) = g(z, γ)
    function prox_g!(v, z, ρ)
        @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)
        v .= soft_threshold.(z, γ/ρ)
    end
    solver = GeNIOS.GenericSolver(
        f, grad_f!, Hf,         # f(x)
        g, prox_g!,             # g(z)
        -I, zeros(p);            # A, c: Ax + z = c
        ρ=1.0, α=1.0
    )
    res = solve!(solver; options=GeNIOS.SolverOptions(relax=true, verbose=false))

    test_optimality_conditions_lasso(solver, 5e-3, A, b)
end

@testset "ML solver" begin
    # # Lasso problem: min 1/2 ||Ax - b||² + γ||x||₁
    f2(x) = 0.5*x^2 
    df2(x) = x
    d2f2(x) = 1.0
    f2conj(x) = 0.5*x^2
    λ1 = γ
    λ2 = 0.0

    Adata = copy(A)
    bdata = copy(b)
    solver2 = GeNIOS.MLSolver(f2, df2, d2f2, λ1, λ2, Adata, bdata; fconj=f2conj)
    res = solve!(solver2; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, tol=1e-3, verbose=false))

    test_optimality_conditions(solver2, 5e-3)

    λ2 = λ1/10
    solver_en = GeNIOS.MLSolver(f2, df2, d2f2, λ1, λ2, Adata, bdata; fconj=f2conj)
    res = solve!(solver_en; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, tol=1e-3, verbose=false))
    
    test_optimality_conditions(solver_en, 5e-3)
end