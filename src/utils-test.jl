# Lasso problem: min 1/2 ||Ax - b||² + γ||x||₁
# Returns A, b, xstar, γmax
#   - γmax is the maximum possible value of γ, above which the solution is zero
function generate_lasso_problem(n, p; rseed=1, sparsity=0.1)
    Random.seed!(rseed)

    # Generate random data matrix A
    A = randn(n, p)
    A .-= sum(A, dims=1) ./ n
    normalize!.(eachcol(A))

    # Generate sparse solution xstar
    xstar = sprandn(p, sparsity)

    # Generate right-hand side b
    b = A*xstar + 1e-3*randn(n)

    # Generate regularization parameter γ
    γmax = norm(A'*b, Inf)

    return A, b, xstar, γmax
end
