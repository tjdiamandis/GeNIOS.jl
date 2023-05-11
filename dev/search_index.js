var documenterSearchIndex = {"docs":
[{"location":"guide/#User-Guide","page":"User Guide","title":"User Guide","text":"","category":"section"},{"location":"guide/#Performance-Tips","page":"User Guide","title":"Performance Tips","text":"","category":"section"},{"location":"guide/#Nyström-PCG-Parameters","page":"User Guide","title":"Nyström PCG Parameters","text":"","category":"section"},{"location":"api/#API-Reference","page":"API reference","title":"API Reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [GeNIOS]","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"The source files for all examples can be found in /examples.","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"EditURL = \"https://github.com/tjdiamandis/GeNIOS.jl/blob/main/examples/huber.jl\"","category":"page"},{"location":"examples/huber/#Huber-Fitting","page":"Huber Fitting","title":"Huber Fitting","text":"","category":"section"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"This example sets up a ell_1-regularized huber fitting problem using the MLSolver interface provided by GeNIOS. Huber fitting is a form of 'robust regression' that is less sensitive to outliers.","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"The huber loss function is defined as","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"f^mathrmhub(x) = begincases\nfrac12x^2  lvert xrvert leq 1 \nx - frac12  lvert x rvert  1\nendcases","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"We want to solve the problem","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"beginarrayll\ntextminimize      sum_i=1^N f^mathrmhub(a_i^T x - b_i) + gamma x_1\nendarray","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"using Pkg\nPkg.activate(joinpath(@__DIR__, \"..\", \"..\"))\n\nusing GeNIOS\nusing Random, LinearAlgebra, SparseArrays","category":"page"},{"location":"examples/huber/#Generating-the-problem-data","page":"Huber Fitting","title":"Generating the problem data","text":"","category":"section"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"Random.seed!(1)\nN, n = 200, 400\nA = randn(N, n)\nA .-= sum(A, dims=1) ./ N\nnormalize!.(eachcol(A))\nxstar = sprandn(n, 0.1)\nb = A*xstar + 1e-3*randn(N)\n\n# Add outliers\nb += 10*collect(sprand(N, 0.05))\nγ = 0.05*norm(A'*b, Inf)","category":"page"},{"location":"examples/huber/#MLSolver-interface","page":"Huber Fitting","title":"MLSolver interface","text":"","category":"section"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"We just need to specify f and the regularization parameters.","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"# Huber problem: min ∑ fʰᵘᵇ(aᵢᵀx - bᵢ) + γ||x||₁\nf(x) = abs(x) <= 1 ? 0.5*x^2 : abs(x) - 0.5\ndf(x) = abs(x) <= 1 ? x : sign(x)\nd2f(x) = abs(x) <= 1 ? 1 : 0\nλ1 = γ\nλ2 = 0.0\nsolver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b)\nres = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=false, verbose=true))\nrmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)\nprintln(\"Final RMSE: $(round(rmse, digits=8))\")","category":"page"},{"location":"examples/huber/#Automatic-differentiation","page":"Huber Fitting","title":"Automatic differentiation","text":"","category":"section"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"We could have let the solver figure out the derivatives for us as well:","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"f(x) = abs(x) <= 1 ? 0.5*x^2 : abs(x) - 0.5\nλ1 = γ\nλ2 = 0.0\nsolver = GeNIOS.MLSolver(f, λ1, λ2, A, b)\nres = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=false, verbose=true))\nrmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)\nprintln(\"Final RMSE: $(round(rmse, digits=8))\")","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"","category":"page"},{"location":"examples/huber/","page":"Huber Fitting","title":"Huber Fitting","text":"This page was generated using Literate.jl.","category":"page"},{"location":"method/#Algorithm","page":"Solution method","title":"Algorithm","text":"","category":"section"},{"location":"method/#ADMM-iteration","page":"Solution method","title":"ADMM iteration","text":"","category":"section"},{"location":"method/#Fast-linear-system-solves-with-Nyström-PCG","page":"Solution method","title":"Fast linear system solves with Nyström PCG","text":"","category":"section"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"The source files for all examples can be found in /examples.","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"EditURL = \"https://github.com/tjdiamandis/GeNIOS.jl/blob/main/examples/portfolio.jl\"","category":"page"},{"location":"examples/portfolio/#Markowitz-Portfolio-Optimization","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"","category":"section"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"This example shows how to solve a Markowitz portfolio optimization problem using both the generic and the QP interface of GeNIOS.","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"Specifically, we want to solve the problem","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"beginarrayll\ntextminimize      (12)gamma x^TSigma x - mu^Tx \ntextsubject to    mathbf1^Tx = 1 \n                     x geq 0\nendarray","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"where Sigma is the covariance matrix of the returns of the assets, mu is the expected return of each asset, and gamma is a risk adversion parameter. The variable x represents the fraction of the total wealth invested in each asset.","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"using GeNIOS\nusing Random, LinearAlgebra, SparseArrays","category":"page"},{"location":"examples/portfolio/#Generating-the-problem-data","page":"Markowitz Portfolio Optimization","title":"Generating the problem data","text":"","category":"section"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"Note that we generate the covariance matrix Sigma as a diagonal plus low-rank matrix, which is a common model for financial data and referred to as a 'factor model'","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"Random.seed!(1)\nk = 5\nn = 100k\n# Σ = F*F' + diag(d)\nF = sprandn(n, k, 0.5)\nd = rand(n) * sqrt(k)\nμ = randn(n)\nγ = 1;\nnothing #hide","category":"page"},{"location":"examples/portfolio/#QPSolver-interface","page":"Markowitz Portfolio Optimization","title":"QPSolver interface","text":"","category":"section"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"The easiest interface for this problem is the QPSolver, where we just need to specify P, q, M, l, and u. The Markowitz portfolio optimization problem is equivalent to the following 'standard form' QP:","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"beginarrayll\ntextminimize      (12)x^T(gamma Sigma) x + (-mu)Tx \ntextsubject to   \nbeginbmatrix1  0endbmatrix\nleq beginbmatrix I  mathbf1^T endbmatrix x\nleq beginbmatrixinfty  1endbmatrix\nendarray","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"P = γ*(F*F' + Diagonal(d))\nq = -μ\nM = vcat(I, ones(1, n))\nl = vcat(zeros(n), ones(1))\nu = vcat(Inf*ones(n), ones(1))\nsolver = GeNIOS.QPSolver(P, q, M, l, u)\nres = solve!(solver; options=GeNIOS.SolverOptions(eps_abs=1e-6))\nprintln(\"Optimal value: $(round(solver.obj_val, digits=4))\")","category":"page"},{"location":"examples/portfolio/#Performance-improvements","page":"Markowitz Portfolio Optimization","title":"Performance improvements","text":"","category":"section"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"We can also define custom operators for P and M to speed up the computation.","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"# P = γ*(F*F' + Diagonal(d))\nstruct FastP\n    F\n    d\n    γ\n    vk\nend\nfunction LinearAlgebra.mul!(y::AbstractVector, P::FastP, x::AbstractVector)\n    mul!(P.vk, P.F', x)\n    mul!(y, P.F, P.vk)\n    @. y += P.d*x\n    @. y *= P.γ\n    return nothing\nend\nP = FastP(F, d, γ, zeros(k))\n\n# M = vcat(I, ones(1, n))\nstruct FastM\n    n::Int\nend\nBase.size(M::FastM) = (M.n+1, M.n)\nBase.size(M::FastM, d::Int) = d <= 2 ? size(M)[d] : 1\nfunction LinearAlgebra.mul!(y::AbstractVector, M::FastM, x::AbstractVector)\n    y[1:M.n] .= x\n    y[end] = sum(x)\n    return nothing\nend\nLinearAlgebra.adjoint(M::FastM) = Adjoint{Float64, FastM}(M)\nfunction LinearAlgebra.mul!(x::AbstractVector{T}, M::Adjoint{T, FastM}, y::AbstractVector{T}) where T <: Number\n    @. x = y[1:M.parent.n] + y[end]\n    return nothing\nend\nfunction LinearAlgebra.mul!(x::AbstractVector{T}, M::Adjoint{T, FastM}, y::AbstractVector{T}, α::T, β::T) where T <: Number\n    @. x = α * ( y[1:M.parent.n] + y[end] ) + β * x\n    return nothing\nend\nM = FastM(n)\nsolver = GeNIOS.QPSolver(P, q, M, l, u);\nres = solve!(solver; options=GeNIOS.SolverOptions(eps_abs=1e-6));\nprintln(\"Optimal value: $(round(solver.obj_val, digits=4))\")","category":"page"},{"location":"examples/portfolio/#GenericSolver-interface","page":"Markowitz Portfolio Optimization","title":"GenericSolver interface","text":"","category":"section"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"The GenericSolver interface is more flexible and allows for some speedups via an alternative problem splitting. We will solve the problem","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"beginarrayll\ntextminimize      (12)gamma x^TSigma x - mu^Tx + I(z) \ntextsubject to    -x + z = 0\nendarray","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"where I(z) is the indicator function for the set z mid z ge 0 text and  mathbf1^Tz = 1. The gradient and Hessian of f(x) = (12)gamma x^TSigma x are easy to compute. The proximal operator of g(z) = I(z) is simply the projection on this set, which can be solved via a one-dimensional root-finding problem (see appendix ?? of [our paper]).","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"# f(x) = γ/2 xᵀ(FFᵀ + D)x - μᵀx\nfunction f(x, F, d, μ, γ, tmp)\n    mul!(tmp, F', x)\n    qf = sum(w->w^2, tmp)\n    qf += sum(i->d[i]*x[i]^2, 1:n)\n\n    return γ/2 * qf - dot(μ, x)\nend\nf(x) = f(x, F, d, μ, γ, zeros(k))\n\n#  ∇f(x) = γ(FFᵀ + D)x - μ\nfunction grad_f!(g, x, F, d, μ, γ, tmp)\n    mul!(tmp, F', x)\n    mul!(g, F, tmp)\n    @. g += d*x\n    @. g *= γ\n    @. g -= μ\n    return nothing\nend\ngrad_f!(g, x) = grad_f!(g, x, F, d, μ, γ, zeros(k))\n\n# ∇²f(x) = γ(FFᵀ + D)\nstruct HessianMarkowitz{T, S <: AbstractMatrix{T}} <: HessianOperator\n    F::S\n    d::Vector{T}\n    vk::Vector{T}\nend\nfunction LinearAlgebra.mul!(y, H::HessianMarkowitz, x)\n    mul!(H.vk, H.F', x)\n    mul!(y, H.F, H.vk)\n    @. y += d*x\n    return nothing\nend\nHf = HessianMarkowitz(F, d, zeros(k))\n\n# g(z) = I(z)\nfunction g(z)\n    T = eltype(z)\n    return all(z .>= zero(T)) && abs(sum(z) - one(T)) < 1e-6 ? 0 : Inf\nend\nfunction prox_g!(v, z, ρ)\n    z_max = maximum(w->abs(w), z)\n    l = -z_max - 1\n    u = z_max\n\n    # bisection search to find zero of F\n    while u - l > 1e-8\n        m = (l + u) / 2\n        if sum(w->max(w - m, zero(eltype(z))), z) - 1 > 0\n            l = m\n        else\n            u = m\n        end\n    end\n    ν = (l + u) / 2\n    @. v = max(z - ν, zero(eltype(z)))\n    return nothing\nend\n\nsolver = GeNIOS.GenericSolver(\n    f, grad_f!, Hf,         # f(x)\n    g, prox_g!,             # g(z)\n    I, zeros(n);           # M, c: Mx + z = c\n)\nres = solve!(solver)\nprintln(\"Optimal value: $(round(solver.obj_val, digits=4))\")","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"","category":"page"},{"location":"examples/portfolio/","page":"Markowitz Portfolio Optimization","title":"Markowitz Portfolio Optimization","text":"This page was generated using Literate.jl.","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"The source files for all examples can be found in /examples.","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"EditURL = \"https://github.com/tjdiamandis/GeNIOS.jl/blob/main/examples/constrained-ls.jl\"","category":"page"},{"location":"examples/constrained-ls/#Constrained-Least-Squares","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"","category":"section"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"This example setsup a constrained least squares problem using the quadratic program interface of GeNIOS.jl. It is from the OSQP docs.","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"Specifically, we want to solve the problem","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"beginarrayll\ntextminimize      Ax - b_2^2 \ntextsubject to    0 leq x leq 1\nendarray","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"using GeNIOS\nusing Random, LinearAlgebra, SparseArrays","category":"page"},{"location":"examples/constrained-ls/#Generating-the-problem-data","page":"Constrained Least Squares","title":"Generating the problem data","text":"","category":"section"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"Random.seed!(1)\nm, n = 30, 20\nAd = sprandn(m, n, 0.7)\nb = randn(m);\nnothing #hide","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"For convenience, we will introdudce a new variable y = Ax - b. The problem becomes","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"beginarrayll\ntextminimize      y^Ty \ntextsubject to    Ax - y = b \n                     0 leq x leq 1\nendarray","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"In OSQP form, this problem is","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"beginarrayll\ntextminimize      y^Ty \ntextsubject to    Ax - y = z_1 \n                     x = z_2 \n                     b leq z_1 leq b \n                     0 leq z_2 leq 1\nendarray","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"or more explicitly","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"beginarrayll\ntextminimize      beginbmatrix x  y endbmatrix^T beginbmatrix 0     I endbmatrix beginbmatrix x  y endbmatrix \ntextsubject to    beginbmatrix A  -I  I  0 endbmatrix beginbmatrix x  y endbmatrix = z\n                     beginbmatrix b  0 endbmatrix leq z leq beginbmatrix b  1 endbmatrix\nendarray","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"P = blockdiag(spzeros(n, n), sparse(I, m, m))\nq = spzeros(n + m)\nM = [\n    Ad                  -sparse(I, m, m);\n    sparse(I, n, n)     spzeros(n, m)\n]\nl = [b; spzeros(n)]\nu = [b; ones(n)];\nnothing #hide","category":"page"},{"location":"examples/constrained-ls/#Building-and-solving-the-problem","page":"Constrained Least Squares","title":"Building and solving the problem","text":"","category":"section"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"Now we setup the solver and solve the problem","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"solver = GeNIOS.QPSolver(P, q, M, l, u)\nres = solve!(\n    solver; options=GeNIOS.SolverOptions(\n        relax=true,\n        max_iters=1000,\n        eps_abs=1e-6,\n        eps_rel=1e-6,\n        verbose=true)\n);\nnothing #hide","category":"page"},{"location":"examples/constrained-ls/#Examining-the-solution","page":"Constrained Least Squares","title":"Examining the solution","text":"","category":"section"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"We can check the solution and its optimality","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"xstar = solver.zk[m+1:end]\nls_residual =  1/√(m) * norm(Ad*xstar - b)\nfeas = all(xstar .>= 0) && all(xstar .<= 1)\nprintln(\"Is feasible? $feas\")\nprintln(\"Least Squres RMSE = $(round(ls_residual, digits=8))\")\nprintln(\"Primal residual: $(round(solver.rp_norm, digits=8))\")\nprintln(\"Dual residual: $(round(solver.rd_norm, digits=8))\")","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"","category":"page"},{"location":"examples/constrained-ls/","page":"Constrained Least Squares","title":"Constrained Least Squares","text":"This page was generated using Literate.jl.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"The source files for all examples can be found in /examples.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"EditURL = \"https://github.com/tjdiamandis/GeNIOS.jl/blob/main/examples/lasso.jl\"","category":"page"},{"location":"examples/lasso/#Lasso","page":"Lasso","title":"Lasso","text":"","category":"section"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"This example sets up a lasso regression problem with three different interfaces provided by GeNIOS.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"Specifically, we want to solve the problem","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"beginarrayll\ntextminimize      (12)Ax - b_2^2 + gamma x_1\nendarray","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"using GeNIOS\nusing Random, LinearAlgebra, SparseArrays","category":"page"},{"location":"examples/lasso/#Generating-the-problem-data","page":"Lasso","title":"Generating the problem data","text":"","category":"section"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"Random.seed!(1)\nm, n = 200, 400\nA = randn(m, n)\nA .-= sum(A, dims=1) ./ m\nnormalize!.(eachcol(A))\nxstar = sprandn(n, 0.1)\nb = A*xstar + 1e-3*randn(m)\nγ = 0.05*norm(A'*b, Inf)","category":"page"},{"location":"examples/lasso/#MLSolver-interface","page":"Lasso","title":"MLSolver interface","text":"","category":"section"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"The easiest interface for this problem is the MLSolver, where we just need to specify f and the regularization parameters","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"f(x) = 0.5*x^2\ndf(x) = x\nd2f(x) = 1.0\nfconj(x) = 0.5*x^2\nλ1 = γ\nλ2 = 0.0\nsolver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b; fconj=fconj)\nres = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, dual_gap_tol=1e-3, verbose=true))\nrmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)\nprintln(\"Final RMSE: $(round(rmse, digits=8))\")","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"Note that we also defined the conjugate function of f, defined as","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"f^*(y) = sup_x yx - f(x)","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"which allows us to use the dual gap as a stopping criterion (see appendix C of our paper for a derivation). Specifying the conjugate function is optional, and the solver will fall back to using the primal and dual residuals if it is not specified.","category":"page"},{"location":"examples/lasso/#Automatic-differentiation","page":"Lasso","title":"Automatic differentiation","text":"","category":"section"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"Note that we can also let the solver use forward-mode automatic differentiation to compute the first and second derivatives of f. The interface is the same, but we drop these arguments:","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"solver = GeNIOS.MLSolver(f, λ1, λ2, A, b; fconj=fconj)\nres = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, dual_gap_tol=1e-3, verbose=true))\nrmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)\nprintln(\"Final RMSE: $(round(rmse, digits=8))\")","category":"page"},{"location":"examples/lasso/#QPSolver-interface","page":"Lasso","title":"QPSolver interface","text":"","category":"section"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"The QPSolver interface requires us to specify the problem in the form","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"beginarrayll\ntextminimize      (12)x^TPx + q^Tx \ntextsubject to     l leq Mx leq u\nendarray","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"We introduce the new variable t and inntroduce the constraint -t leq x leq t to enforce the ell_1 norm. The problem then becomes","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"beginarrayll\ntextminimize      (12)x^TA^TAx + b^TAx + gamma mathbf1^Tt \ntextsubject to   \nbeginbmatrix0  -inftyendbmatrix\nleq beginbmatrix I  I  I  -I endbmatrix beginbmatrixx  tendbmatrix\nleq beginbmatrixinfty  0endbmatrix\nendarray","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"P = blockdiag(sparse(A'*A), spzeros(n, n))\nq = vcat(-A'*b, γ*ones(n))\nM = [\n    sparse(I, n, n)     sparse(I, n, n);\n    sparse(I, n, n)     -sparse(I, n, n)\n]\nl = [zeros(n); -Inf*ones(n)]\nu = [Inf*ones(n); zeros(n)]\nsolver = GeNIOS.QPSolver(P, q, M, l, u)\nres = solve!(\n    solver; options=GeNIOS.SolverOptions(\n        relax=true,\n        max_iters=1000,\n        eps_abs=1e-4,\n        eps_rel=1e-4,\n        verbose=true)\n);\nrmse = sqrt(1/m*norm(A*solver.xk[1:n] - b, 2)^2)\nprintln(\"Final RMSE: $(round(rmse, digits=8))\")","category":"page"},{"location":"examples/lasso/#Generic-interface","page":"Lasso","title":"Generic interface","text":"","category":"section"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"Finally, we use the generic interface, which provides a large amount of control but is more complicated to use than the specialized interfaces demonstrated above.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"First, we define the custom HessianOperator, which is used to solve the linear system in the x-subproblem. Since the Hessian of the objective is simply A^TA, this operator is simple for the Lasso problem. However, note that, since this is a custom type, we can speed up the multiplication to be more efficient than if we were to use A^TA directly.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"The update! function is called before the solution of the x-subproblem to update the Hessian, if necessary.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"struct HessianLasso{T, S <: AbstractMatrix{T}} <: HessianOperator\n    A::S\n    vm::Vector{T}\nend\nfunction LinearAlgebra.mul!(y, H::HessianLasso, x)\n    mul!(H.vm, H.A, x)\n    mul!(y, H.A', H.vm)\n    return nothing\nend\n\nfunction update!(::HessianLasso, ::Solver)\n    return nothing\nend","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"Now, we define f, its gradient, g, and its proximal operator.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"function f(x, A, b, tmp)\n    mul!(tmp, A, x)\n    @. tmp -= b\n    return 0.5 * sum(w->w^2, tmp)\nend\nf(x) = f(x, A, b, zeros(m))\nfunction grad_f!(g, x, A, b, tmp)\n    mul!(tmp, A, x)\n    @. tmp -= b\n    mul!(g, A', tmp)\n    return nothing\nend\ngrad_f!(g, x) = grad_f!(g, x, A, b, zeros(m))\nHf = HessianLasso(A, zeros(m))\ng(z, γ) = γ*sum(x->abs(x), z)\ng(z) = g(z, γ)\nfunction prox_g!(v, z, ρ)\n    @inline soft_threshold(x::T, κ::T) where {T <: Real} = sign(x) * max(zero(T), abs(x) - κ)\n    v .= soft_threshold.(z, γ/ρ)\nend","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"Finally, we can solve the problem.","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"solver = GeNIOS.GenericSolver(\n    f, grad_f!, Hf,         # f(x)\n    g, prox_g!,             # g(z)\n    I, zeros(n);           # M, c: Mx + z = c\n    ρ=1.0, α=1.0\n)\nres = solve!(solver; options=GeNIOS.SolverOptions(relax=true, verbose=true))\nrmse = sqrt(1/m*norm(A*solver.zk - b, 2)^2)\nprintln(\"Final RMSE: $(round(rmse, digits=8))\")","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"","category":"page"},{"location":"examples/lasso/","page":"Lasso","title":"Lasso","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = GeNIOS","category":"page"},{"location":"#GeNIOS.jl","page":"Home","title":"GeNIOS.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TODO","category":"page"},{"location":"#Documentation-Contents:","page":"Home","title":"Documentation Contents:","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\", \"method.md\", \"guide.md\"]\nDepth = 1","category":"page"},{"location":"#Examples:","page":"Home","title":"Examples:","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"examples/constrained-ls.md\",\n    \"examples/huber.md\",\n    \"examples/lasso.md\",\n    \"examples/logistic.md\",\n    \"examples/portfolio.md\"\n]\nDepth = 1","category":"page"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GeNIOS solves convex optimization problems of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginarrayll\ntextminimize      f(x) + g(z) \ntextsubject to    Mx + z = c\nendarray","category":"page"},{"location":"","page":"Home","title":"Home","text":"where x in mathbbR^n and z in mathbbR^m are the optimization variables. The function f is assumed to be smooth, with known first and second derivatives, and the function g must have a known proximal operator.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Compared to conic form programs, the form we use in GeNIOS facilitates custom  subroutines that often provide significant speedups. To ameliorate the extra  complexity, we provide a few interfaces that take advantage of special problem structure.","category":"page"},{"location":"#Interfaces","page":"Home","title":"Interfaces","text":"","category":"section"},{"location":"#QP-interface","page":"Home","title":"QP interface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Many important problems in machine learning, finance, operations research, and control can be formulated as QPs. Examples include","category":"page"},{"location":"","page":"Home","title":"Home","text":"Lasso regression\nPortfolio optimization\nTrajectory optimization\nModel predictive control\nAnd many more...","category":"page"},{"location":"","page":"Home","title":"Home","text":"GeNIOS accepts QPs of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginarrayll\ntextminimize      (12)x^TPx + q^Tx \ntextsubject to    l leq Mx leq u\nendarray","category":"page"},{"location":"","page":"Home","title":"Home","text":"which can be constructed using","category":"page"},{"location":"","page":"Home","title":"Home","text":"solver = GeNIOS.QPSolver(P, q, M, l, u)","category":"page"},{"location":"#ML-interface","page":"Home","title":"ML interface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In machine learning problems, we can take advantage of additional structure. In our MLSolver, we assume the problem is of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginarrayll\ntextminimize      sum_i=1^m f(a_i^Tx - b_i) + (12)lambda_2x_2^2 + lambda_1x_1\nendarray","category":"page"},{"location":"","page":"Home","title":"Home","text":"where f is the per-sample loss function. Let A be a matrix with rows a_i and b be a vector with entries b_i. The MLSolver is constructed via","category":"page"},{"location":"","page":"Home","title":"Home","text":"solver = GeNIOS.MLSolver(f, df, d2f, λ1, λ2, A, b)","category":"page"},{"location":"","page":"Home","title":"Home","text":"where df, and d2f are scalar functions giving the first and second derivative of f respectively.","category":"page"},{"location":"","page":"Home","title":"Home","text":"In the future, we may extend this interface to allow constraints on x, but for now, you can use the generic interface to specify constraints in machine learning problems with non-quadratic objective functions.","category":"page"},{"location":"#Generic-interface","page":"Home","title":"Generic interface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"For power users, we expose a fully generic interface for the problem","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginarrayll\ntextminimize      f(x) + g(z) \ntextsubject to    Mx + z = c\nendarray","category":"page"},{"location":"","page":"Home","title":"Home","text":"Users must specify f, nabla f, nabla^2 f (via a HessianOperator),  g, the proximal operator of g, M, and c. We provide full details of these functions in the User Guide. Also checkout the Lasso example to  see a problem written in all three ways.","category":"page"},{"location":"#Algorithm","page":"Home","title":"Algorithm","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GeNIOS follows a similar approach to OSQP [1], solving the convex optimization problem using ADMM [2]. Note that the problem form of GeNIOS is, however, more  general than conic form solvers. The key algorithmic differences lies in the  x-subproblem of ADMM. Instead of solving this problem exactly, GeNIOS solves a quadratic approximation to this problem. Recent work [3] shows that this  approximation does not harm ADMM's convergence rate. The subproblem is then solved via the iterative conjugate gradient method with a randomized preconditioner [4]. GeNIOS also incorporates several other heuristics from the literature. Please see our paper for additional details.","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The JuMP interface is the easiest way to use GeNIOS. A simple Markowitz portfolio example is below.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using JuMP, GeNIOS\n# TODO:","category":"page"},{"location":"","page":"Home","title":"Home","text":"However, the native interfaces can be called directly by specifying the problem data. Using the QPSolver, is it written as","category":"page"},{"location":"","page":"Home","title":"Home","text":"# TODO:","category":"page"},{"location":"","page":"Home","title":"Home","text":"And, finally, using the fully general interface, it is written as","category":"page"},{"location":"","page":"Home","title":"Home","text":"# TODO:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Please see the User Guide for a full explanation of the interfaces, keyword  arguments and performance tips. Also check out the examples as well.","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"[1]: Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"[2]: Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[3]: Frangella, Z., Zhao, S., Diamandis, T., Stellato, B., & Udell, M. (2023). On the (linear) convergence of Generalized Newton Inexact ADMM.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[4]: Frangella, Z., Tropp, J. A., & Udell, M. (2021). Randomized Nyström Preconditioning.","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"The source files for all examples can be found in /examples.","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"EditURL = \"https://github.com/tjdiamandis/GeNIOS.jl/blob/main/examples/logistic.jl\"","category":"page"},{"location":"examples/logistic/#Logistic-Regression","page":"Logistic Regression","title":"Logistic Regression","text":"","category":"section"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"This example sets up a ell_1-regularized logistic regression problem using the MLSolver interface provided by GeNIOS.","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"Specifically, we want to solve the problem","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"beginarrayll\ntextminimize      sum_i=1^N log(1 + exp(a_i^T x)) + gamma x_1\nendarray","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"using GeNIOS\nusing Random, LinearAlgebra, SparseArrays","category":"page"},{"location":"examples/logistic/#Generating-the-problem-data","page":"Logistic Regression","title":"Generating the problem data","text":"","category":"section"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"Random.seed!(1)\nN, n = 2_000, 4_000\nÃ = sprandn(N, n, 0.2)\n@views [normalize!(Ã[:, i]) for i in 1:n-1]\nÃ[:,n] .= 1.0\n\nxstar = zeros(n)\ninds = randperm(n)[1:100]\nxstar[inds] .= randn(length(inds))\nb̃ = sign.(Ã*xstar + 1e-1 * randn(N))\nb = zeros(N)\nA = Diagonal(b̃) * Ã\n\nγmax = norm(0.5*A'*ones(N), Inf)\nγ = 0.05*γmax","category":"page"},{"location":"examples/logistic/#MLSolver-interface","page":"Logistic Regression","title":"MLSolver interface","text":"","category":"section"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"We just need to specify f and the regularization parameters. We also define the conjugate function f^*, defined as","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"f^*(y) = sup_x yx - f(x)","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"to use the dual gap convergence criterion.","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"# Logistic problem: min ∑ log(1 + exp(aᵢᵀx)) + γ||x||₁\nf2(x) = GeNIOS.log1pexp(x)\ndf2(x) = GeNIOS.logistic(x)\nd2f2(x) = GeNIOS.logistic(x) / GeNIOS.log1pexp(x)\nf2conj(x::T) where {T} = (one(T) - x) * log(one(T) - x) + x * log(x)\nλ1 = γ\nλ2 = 0.0\nsolver = GeNIOS.MLSolver(f2, df2, d2f2, λ1, λ2, A, b; fconj=f2conj)\nres = solve!(solver; options=GeNIOS.SolverOptions(relax=true, use_dual_gap=true, dual_gap_tol=1e-4, verbose=true))","category":"page"},{"location":"examples/logistic/#Results","page":"Logistic Regression","title":"Results","text":"","category":"section"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"rmse = sqrt(1/N*norm(A*solver.zk - b, 2)^2)\nprintln(\"Final RMSE: $(round(rmse, digits=8))\")\nprintln(\"Dual gap: $(round(res.dual_gap, digits=8))\")","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"","category":"page"},{"location":"examples/logistic/","page":"Logistic Regression","title":"Logistic Regression","text":"This page was generated using Literate.jl.","category":"page"}]
}
