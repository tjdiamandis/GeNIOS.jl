function update_x!(
    solver::MLSolver,
    options::SolverOptions,
    )

    # if options.logging, time this (cataloged as 'linear system solve')
    if options.logging
        time_start = time_ns()
    end
    @inline function fg!(f, g, x, solver::MLSolver{T}) where {T}

        # compute Ax - b
        mul!(solver.pred, solver.data.Adata, x)
        solver.pred .-= solver.data.bdata
    
        # compute vector Mx - zᵏ - c + uᵏ
        mul!(solver.cache.vn, solver.data.M, x)
        @. solver.cache.vn += -solver.zk - solver.data.c + solver.uk
    
        if !isnothing(g)
            # compute ∇f(x)
            @. solver.cache.vN = solver.data.df(solver.pred)
            mul!(g, solver.data.Adata', solver.cache.vN)
    
            # g = ∇f(x) + ρMᵀ(Mx - zᵏ - c + uᵏ)
            mul!(g, solver.data.M', solver.cache.vn, solver.ρ, one(T))
        end
    
        if !isnothing(f)
            return sum(solver.data.f, solver.pred) + solver.ρ/2 * sum(abs2, solver.cache.vn)
        end
    end
    @inline fg!(f, g, x) = fg!(f, g, x, solver)
    
    res = Optim.optimize(Optim.only_fg!(fg!), solver.xk, method = Optim.LBFGS())

    solver.xk .= Optim.minimizer(res)

    return options.logging ? (time_ns() - time_start) / 1e9 : nothing

end