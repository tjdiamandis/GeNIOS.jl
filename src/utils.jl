# --- Solver printing utils ---
function print_header(format, data)
    @printf(
        "\n──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
end


function print_footer()
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
    )
end


function print_iter_func(format, data)
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
end

function print_format(::Solver, ::SolverOptions)
    return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s"]
end

function print_headers(::Solver, ::SolverOptions)
    return ["Iteration", "Obj Val", "r_primal", "r_dual", "ρ", "Time"]
end

function iter_format(::Solver, ::SolverOptions)
    return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
end

function iter_data(solver::Solver, ::SolverOptions, t, time_sec)
    return (
        string(t),
        solver.obj_val,
        solver.rp_norm,
        solver.rd_norm,
        solver.ρ,
        time_sec
    )
end

function print_format(::MLSolver, options::SolverOptions)
    if options.use_dual_gap
        return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s"]
    else
        return ["%13s", "%14s", "%14s", "%14s", "%14s", "%14s", "%14s"]
    end
end

function print_headers(::MLSolver, options::SolverOptions)
    if options.use_dual_gap
        return ["Iteration", "Objective", "RMSE", "Dual Gap", "r_primal", "r_dual", "ρ", "Time"]
    else
        return ["Iteration", "Objective", "RMSE", "r_primal", "r_dual", "ρ", "Time"]
    end
end

function iter_format(::MLSolver, options::SolverOptions)
    if options.use_dual_gap
        return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
    else
        return ["%13s", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%14.3e", "%13.3f"]
    end
end

function iter_data(solver::MLSolver, options::SolverOptions, t, time_sec)
    if options.use_dual_gap
        return (
            string(t),
            solver.obj_val,
            sqrt(solver.loss / solver.data.N),
            solver.dual_gap,
            solver.rp_norm,
            solver.rd_norm,
            solver.ρ,
            time_sec
        )
    else
        return (
            string(t),
            solver.obj_val,
            sqrt(solver.loss / solver.data.N),
            solver.rp_norm,
            solver.rd_norm,
            solver.ρ,
            time_sec
        )
    end
end

# Generates dual form:
#   min c^Tx st ΣF_ix_i + G ∈ PSDCONE()
#   Build from KKT conditions:
#       Fx = sum(F[i]*xstar[i] for i in 1:n) + G
#       all(eigvals(Matrix(Fx)) .>= 0)
#       all(eigvals(D) .>= 0)
#       all([0 .== c[i] - tr(F[i]*D) for i in 1:n])
#       tr(Fx*D) .== 0
"""
    generate_random_sdp(n; rand_seed=0)

Generates a random dual form SDP with side dimension `n`:
`min c'*x s.t. sum(F[i]*x[i]) + G ⪰ 0`

Returns `c, F, G, xstar, D`, where `xstar` and `D` are optimal primal and dual
variables respectively
"""
function generate_random_sdp(n; rand_seed=0)
    Random.seed!(rand_seed)

    D = diagm(1 .+ rand(n))
    F = Vector{SparseMatrixCSC{Float64, Int}}(undef, n)
    c = Vector{Float64}(undef, n)
    for i in 1:n
        F[i] = spzeros(n, n)
        block_size = randn() < 1.5 ? 2 : n ÷ 20
        F[i][i:min(i+block_size,n), i:min(i+block_size,n)] .= 1
        c[i] = tr(D*F[i])
    end
    xstar = rand(n)
    Fx = sum(F[i]*xstar[i] for i in 1:n)
    G = -Fx

    return c, F, G, xstar, D
end

"""
   unvec_symm(x)

Returns a dim-by-dim symmetric matrix corresponding to `x`.
`x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric matrix
```
X = [ X11     X12/√2 ... X1k/√2
      X21/√2  X22    ... X2k/√2
      ...
      Xk1/√2  Xk2/√2 ... Xkk ],
```
where
`vec(X = (X11, X12, X22, X13, X23, ..., Xkk)`

Note that the factor √2 preserves inner products:
`x'*c = Tr(unvec_symm(c, dim) * unvec_symm(x, dim))`
"""
function unvec_symm(x::Vector{T}, dim::Int) where {T <: Number}
    X = zeros(T, dim, dim)
    idx = 1
    for i in 1:dim
        for j in 1:i
            if i == j
                X[i,j] = x[idx]
            else
                X[j,i] = X[i,j] = x[idx] / sqrt(2)
            end
            idx += 1
        end
    end
    return X
end


function unvec_symm(x::Vector{T}) where {T <: Number}
    dim = Int( (-1 + sqrt(1 + 8*length(x))) / 2 )
    dim * (dim + 1) ÷ 2 != length(x) && throw(DomainError("invalid vector length"))
    return unvec_symm(x, dim)
end


function unvec_symm!(X::Matrix{T}, x::Vector{T}, dim::Int) where {T <: Number}
    idx = 1
    for i in 1:dim
        for j in 1:i
            if i == j
                X[i,j] = x[idx]
            else
                X[j,i] = X[i,j] = x[idx] / sqrt(2)
            end
            idx += 1
        end
    end
    return nothing
end

function unvec_symm!(X::Matrix{T}, x::Vector{T}) where {T <: Number}
    dim = Int( (-1 + sqrt(1 + 8*length(x))) / 2 )
    dim * (dim + 1) ÷ 2 != length(x) && throw(DomainError("invalid vector length"))
    return unvec_symm!(X, x, dim)
end


"""
   vec_symm(X)

Returns a vectorized representation of a symmetric matrix `X`.
`vec(X) = (X11, √2*X12, X22, √2*X13, X23, ..., Xkk)`

Note that the factor √2 preserves inner products:
`x'*c = Tr(unvec_symm(c, dim) * unvec_symm(x, dim))`
"""
function vec_symm(X)
    x_vec = sqrt(2).*X[LinearAlgebra.triu(trues(size(X)))]
    idx = 1
    for i in 1:size(X)[1]
        x_vec[idx] =  x_vec[idx]/sqrt(2)
        idx += i + 1
    end
    return x_vec
end

function vec_symm!(x_vec, X, upper_inds)
    @. x_vec = sqrt(2) * @view X[upper_inds]
    idx = 1
    for i in 1:size(X)[1]
        x_vec[idx] =  x_vec[idx]/sqrt(2)
        idx += i + 1
    end
    return x_vec
end