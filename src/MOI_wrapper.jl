import MathOptInterface as MOI

const MOIU = MOI.Utilities
const SparseTriplets = Tuple{Vector{Int},Vector{Int},Vector{<:Any}}
const Affine = MOI.ScalarAffineFunction{Float64}
const Quadratic = MOI.ScalarQuadraticFunction{Float64}
const VectorAffine = MOI.VectorAffineFunction{Float64}

const Interval = MOI.Interval{Float64}
const LessThan = MOI.LessThan{Float64}
const GreaterThan = MOI.GreaterThan{Float64}
const EqualTo = MOI.EqualTo{Float64}
const IntervalConvertible = Union{Interval,LessThan,GreaterThan,EqualTo}

const Zeros = MOI.Zeros
const Nonnegatives = MOI.Nonnegatives
const Nonpositives = MOI.Nonpositives
const SupportedVectorSets = Union{Zeros,Nonnegatives,Nonpositives}

const SPARSE_THRESHOLD_FACTOR = 0.1

# XXX: support other sets directly in conic
# XXX: change to support AbstractFloat
lower(::MOI.Zeros, i::Int) = 0.0
lower(::MOI.Nonnegatives, i::Int) = 0.0
lower(::MOI.Nonpositives, i::Int) = -Inf
upper(::MOI.Zeros, i::Int) = 0.0
upper(::MOI.Nonnegatives, i::Int) = Inf
upper(::MOI.Nonpositives, i::Int) = 0.0

"""
    Optimizer()

Create a new GeNIOS optimizer.
"""
mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    inner::Union{Nothing, GeNIOS.ConicSolver{T}}
    has_result::Bool
    result::Union{Nothing, GeNIOS.GeNIOSResult{T}} #XXX: why nothing?
    is_empty::Bool
    options::GeNIOS.SolverOptions{T}
    sense::MOI.OptimizationSense
    objective_constant::T
    constraint_constant::Vector{T} ### TODO: remove when conic
    rowranges::Dict{Int, UnitRange{Int}}

    # XXX TODO: ???
    # cones::Union{Nothing,Cones{T}}
    # data::Union{Nothing,Data{T}}}
    # cache::Union{Nothing,Cache{T}}

    function Optimizer{T}(; kwargs...) where {T}
        inner = nothing
        has_result = false
        result = nothing
        is_empty = true
        options = GeNIOS.SolverOptions()
        sense = MOI.MIN_SENSE
        objective_constant = zero(T)
        constraint_constant = zeros(T, 0) ### TODO: remove when conic
        rowranges = Dict{Int, UnitRange{Int}}()

        return new{T}(
            inner,
            has_result,
            result,
            is_empty,
            options,
            sense,
            objective_constant,
            constraint_constant,
            rowranges,
        )
    end
end
Optimizer() = Optimizer{Float64}()

# ------------------------------------------------------------------------------
# empty
# ------------------------------------------------------------------------------
function MOI.is_empty(optimizer::Optimizer)
    return optimizer.is_empty
end

# NOTE: does not change SolverOptions()
function MOI.empty!(optimizer::Optimizer{T}) where {T}
    optimizer.inner = nothing
    optimizer.has_result = false
    optimizer.result = nothing
    optimizer.is_empty = true
    optimizer.sense = MOI.MIN_SENSE
    optimizer.objective_constant = zero(T)
    optimizer.constraint_constant = T[] ### TODO: remove when conic
    empty!(optimizer.rowranges)
    return optimizer
end


# ------------------------------------------------------------------------------
#   Attributes
# ------------------------------------------------------------------------------
# SolverName
MOI.get(::Optimizer, ::MOI.SolverName) = "GeNIOS"

# SolverVersion
MOI.get(::Optimizer, ::MOI.SolverVersion) = "v0.0.1"

# RawSolver
MOI.get(optimizer::Optimizer, ::MOI.RawSolver) = optimizer.inner

# RawOptimizerAttribute
function MOI.supports(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return hasfield(GeNIOS.SolverOptions, Symbol(param.name))
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    !MOI.supports(optimizer, param) && throw(MOI.UnsupportedAttribute(param))

    setfield!(optimizer.options, Symbol(param.name), value)
    return nothing
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    !MOI.supports(optimizer, param) && throw(MOI.UnsupportedAttribute(param))

    return getfield(optimizer.options, Symbol(param.name))
end

# Name
MOI.supports(::Optimizer, ::MOI.Name) = false

# Silent
MOI.supports(::Optimizer, ::MOI.Silent) = true
function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("verbose"), !value)
    return nothing
end
MOI.get(optimizer::Optimizer, ::MOI.Silent) = 
    !MOI.get(optimizer, MOI.RawOptimizerAttribute("verbose"))

# TimeLimitSec
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.set(optimizer::Optimizer{T}, ::MOI.TimeLimitSec, value) where {T <: AbstractFloat}
    MOI.set(optimizer, MOI.RawOptimizerAttribute("max_time_sec"), T(value))
    return nothing
end
MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec) = 
    MOI.get(optimizer, MOI.RawOptimizerAttribute("max_time_sec"))

# AbsoluteGapTolerance
MOI.supports(::Optimizer, ::MOI.AbsoluteGapTolerance) = true
function MOI.set(optimizer::Optimizer{T}, ::MOI.AbsoluteGapTolerance, value) where {T <: AbstractFloat}
    MOI.set(optimizer, MOI.RawOptimizerAttribute("eps_abs"), T(value))
    return nothing
end
MOI.get(optimizer::Optimizer, ::MOI.AbsoluteGapTolerance) = 
    MOI.get(optimizer, MOI.RawOptimizerAttribute("eps_abs"))

# RelativeGapTolerance
MOI.supports(::Optimizer, ::MOI.RelativeGapTolerance) = true
function MOI.set(optimizer::Optimizer{T}, ::MOI.RelativeGapTolerance, value) where {T <: AbstractFloat}
    MOI.set(optimizer, MOI.RawOptimizerAttribute("eps_rel"), T(value))
    return nothing
end
MOI.get(optimizer::Optimizer, ::MOI.RelativeGapTolerance) = 
    MOI.get(optimizer, MOI.RawOptimizerAttribute("eps_rel"))

# NumberOfThreads
# TODO: support this (simple change to solver)
MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false


# ------------------------------------------------------------------------------
#   Supported constraints
# ------------------------------------------------------------------------------
# XXX: need Optimizer{T} vs Optimizer??
function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{<:IntervalConvertible},
) where T
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VectorAffineFunction{T}},
    ::Type{<:SupportedVectorSets},
) where T
    return true
end


# ------------------------------------------------------------------------------
#   Non-incremental solve interface
# ------------------------------------------------------------------------------
function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike; copy_names=false) where {T}
    copy_names && error("Copying names is not supported.")
    MOI.empty!(dest)
    
    index_map = MOIU.IndexMap(dest, src)
    assign_constraint_row_ranges!(dest.rowranges, index_map, src)

    obj_sense, P, q, dest.objective_constant = processobjective(src, index_map)

    # TODO: can remove constraint_constant when switch to conic
    M, l, u, dest.constraint_constant = processconstraints(src, index_map, dest.rowranges)

    dest.inner = GeNIOS.QPSolver(P, q, M, l, u; σ=1e-8)
    dest.is_empty = false 
    dest.sense = obj_sense

    # TODO: warm start
    # dest.warmstartcache = WarmStartCache{Float64}(size(A, 2), size(A, 1))
    # processprimalstart!(dest.warmstartcache.x, src, idxmap)
    # processdualstart!(dest.warmstartcache.y, src, idxmap, dest.rowranges)

    return index_map
end


function MOI.optimize!(optimizer::Optimizer)
    optimizer.result = GeNIOS.solve!(optimizer.inner; options=optimizer.options)
    optimizer.has_result = true
    return nothing
end


# Set up index map from `src` variables and constraints to `dest` variables and constraints.
function MOIU.IndexMap(dest::Optimizer, src::MOI.ModelLike)
    idxmap = MOIU.IndexMap()
    var_inds_src = MOI.get(src, MOI.ListOfVariableIndices())
    for i in eachindex(var_inds_src)
        idxmap[var_inds_src[i]] = MOI.VariableIndex(i)
    end

    i = 0
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(dest, F, S) || throw(MOI.UnsupportedConstraint{F, S}())
        con_inds_src = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        for ci in con_inds_src
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{F, S}(i)
        end
    end
    return idxmap
end


function assign_constraint_row_ranges!(
    rowranges::Dict{Int,UnitRange{Int}},
    idxmap::MOIU.IndexMap,
    src::MOI.ModelLike,
)
    startrow = 1
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        cis_src = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        for ci_src in cis_src
            set = MOI.get(src, MOI.ConstraintSet(), ci_src)
            ci_dest = idxmap[ci_src]
            endrow = startrow + MOI.dimension(set) - 1
            rowranges[ci_dest.value] = startrow:endrow
            startrow = endrow + 1
        end
    end
    return nothing
end

# ---- OBJECTIVE ----
# Return objective sense, as well as matrix `P`, vector `q`, and scalar `r` such that objective function is `1/2 x' P x + q' x + r`.
function processobjective(src::MOI.ModelLike, idxmap)
    sense = MOI.get(src, MOI.ObjectiveSense())
    n = MOI.get(src, MOI.NumberOfVariables())
    q = zeros(n)
    if sense != MOI.FEASIBILITY_SENSE
        function_type = MOI.get(src, MOI.ObjectiveFunctionType())
        if function_type == MOI.ScalarAffineFunction{Float64}
            faffine = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
            P = spzeros(n, n)
            processlinearterms!(q, faffine.terms, idxmap)
            r = faffine.constant
        elseif function_type == MOI.ScalarQuadraticFunction{Float64}
            fquadratic = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
            I = [
                Int(idxmap[term.variable_1].value) for
                term in fquadratic.quadratic_terms
            ]
            J = [
                Int(idxmap[term.variable_2].value) for
                term in fquadratic.quadratic_terms
            ]
            V = [term.coefficient for term in fquadratic.quadratic_terms]
            symmetrize!(I, J, V)
            P = sparse(I, J, V, n, n)
            if nnz(P) ≥ SPARSE_THRESHOLD_FACTOR * n^2
                P = Matrix(P)
            end
            processlinearterms!(q, fquadratic.affine_terms, idxmap)
            r = fquadratic.constant
        else
            throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{function_type}()))
        end
        
        if sense == MOI.MAX_SENSE 
            rmul!(P, -1)
            rmul!(q, -1)
            r = -r
        end
    else
        P = spzeros(n, n)
        q = zeros(n)
        r = 0.
    end
    sense, P, q, r
end

function processlinearterms!(
    q,
    terms::Vector{<:MOI.ScalarAffineTerm},
    idxmapfun::Function = identity,
)
    q .= 0
    for term in terms
        var = term.variable
        coeff = term.coefficient
        q[idxmapfun(var).value] += coeff
    end
end

function processlinearterms!(
    q,
    terms::Vector{<:MOI.ScalarAffineTerm},
    idxmap::MOIU.IndexMap,
)
    processlinearterms!(q, terms, var -> idxmap[var])
end

function symmetrize!(I::Vector{Int}, J::Vector{Int}, V::Vector)
    n = length(V)
    (length(I) == length(J) == n) || error()
    for i = 1 : n
        if I[i] != J[i]
            push!(I, J[i])
            push!(J, I[i])
            push!(V, V[i])
        end
    end
end


# ---- CONSTRAINTS ----
function constraint_rows(
    rowranges::Dict{Int, UnitRange{Int}},
    ci::MOI.ConstraintIndex{<:Any, <:MOI.AbstractScalarSet}
)
    rowrange = rowranges[ci.value]
    length(rowrange) == 1 || error()
    return first(rowrange)
end
function constraint_rows(
    rowranges::Dict{Int, UnitRange{Int}}, 
    ci::MOI.ConstraintIndex{<:Any, <:MOI.AbstractVectorSet}
)
    return rowranges[ci.value]
end
constraint_rows(optimizer::Optimizer, ci::MOI.ConstraintIndex) = constraint_rows(optimizer.rowranges, ci)

function processconstraints(
    src::MOI.ModelLike,
    idxmap,
    rowranges::Dict{Int,UnitRange{Int}},
)
    m = mapreduce(length, +, values(rowranges), init=0)
    l = Vector{Float64}(undef, m)
    u = Vector{Float64}(undef, m)
    
    bounds = (l, u)
    constant = Vector{Float64}(undef, m)
    I, J, V = Int[], Int[], Float64[]
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        processconstraints!((I, J, V), bounds, constant, src, idxmap, rowranges, F, S)
    end

    l .-= constant
    u .-= constant
    n = MOI.get(src, MOI.NumberOfVariables())
    M = sparse(I, J, V, m, n)
    
    return (M, l, u, constant)
end

function processconstraints!(
    triplets::SparseTriplets,
    bounds::Tuple{<:Vector,<:Vector},
    constant::Vector{Float64},
    src::MOI.ModelLike,
    idxmap,
    rowranges::Dict{Int,UnitRange{Int}},
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    con_inds_src = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    for ci in con_inds_src
        con_set = MOI.get(src, MOI.ConstraintSet(), ci)
        con_func = MOI.get(src, MOI.ConstraintFunction(), ci)
        rows = constraint_rows(rowranges, idxmap[ci])
        
        processconstant!(constant, rows, con_func)
        processlinearpart!(triplets, con_func, rows, idxmap)
        processconstraintset!(bounds, rows, con_set)
    end
    return nothing
end

function processconstant!(c::Vector{Float64}, row::Int, f::Affine)
    c[row] = MOI.constant(f, Float64)
    return nothing
end

function processconstant!(
    c::Vector{Float64},
    rows::UnitRange{Int},
    f::VectorAffine,
)
    for (i, row) in enumerate(rows)
        c[row] = f.constants[i]
    end
end

function processlinearpart!(
    triplets::SparseTriplets,
    f::MOI.ScalarAffineFunction,
    row::Int,
    idxmap,
)
    (I, J, V) = triplets
    for term in f.terms
        var = term.variable
        coeff = term.coefficient
        col = idxmap[var].value
        push!(I, row)
        push!(J, col)
        push!(V, coeff)
    end
    return nothing
end

function processlinearpart!(
    triplets::SparseTriplets,
    f::MOI.VectorAffineFunction,
    rows::UnitRange{Int},
    idxmap,
)
    (I, J, V) = triplets
    for term in f.terms
        row = rows[term.output_index]
        var = term.scalar_term.variable
        coeff = term.scalar_term.coefficient
        col = idxmap[var].value
        push!(I, row)
        push!(J, col)
        push!(V, coeff)
    end
end

function processconstraintset!(
    bounds::Tuple{<:Vector, <:Vector}, 
    row::Int, 
    s::IntervalConvertible
)
    processconstraintset!(bounds, row, MOI.Interval(s))
end

function processconstraintset!(
    bounds::Tuple{<:Vector, <:Vector}, 
    row::Int, 
    interval::Interval
)
    l, u = bounds
    l[row] = interval.lower
    u[row] = interval.upper
    return nothing
end

function processconstraintset!(
    bounds::Tuple{<:Vector, <:Vector}, 
    rows::UnitRange{Int}, 
    s::S
) where {S<:SupportedVectorSets}
    l, u = bounds
    for (i, row) in enumerate(rows)
        l[row] = lower(s, i)
        u[row] = upper(s, i)
    end
    return nothing
end

MOI.supports(::Optimizer, ::MOI.VariableName) = false
MOI.supports(::Optimizer, ::MOI.ConstraintName) = false


# ------------------------------------------------------------------------------
#   Solution status
# ------------------------------------------------------------------------------
# RawStatusString
function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return string(optimizer.result.status)
end

# TerminationStatus
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    optimizer.has_result || return MOI.OPTIMIZE_NOT_CALLED
    opt_status = optimizer.result.status
    if opt_status == :ITERATION_LIMIT
        return MOI.ITERATION_LIMIT
    elseif opt_status == :TIME_LIMIT
        return MOI.TIME_LIMIT
    elseif opt_status == :INFEASIBLE
        return MOI.INFEASIBLE
    elseif opt_status == :DUAL_INFEASIBLE
        return MOI.INFEASIBLE_OR_UNBOUNDED
    elseif opt_status == :OPTIMAL
        return MOI.OPTIMAL
    else
        error("Unexpected solver status")
        return MOI.INVALID_MODEL
    end
end


# XXX: TODO: Infeasibility certificates
# PrimalStatus
function MOI.get(optimizer::Optimizer, a::MOI.PrimalStatus)
    if a.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    optimizer.has_result || return MOI.NO_SOLUTION
    
    opt_status = optimizer.result.status
    if opt_status == :ITERATION_LIMIT
        return MOI.ITERATION_LIMIT
    elseif opt_status ∈ (:TIME_LIMIT, :ITERATION_LIMIT)
        return MOI.NO_SOLUTION
    elseif opt_status == :INFEASIBLE
        return MOI.INFEASIBLE_POINT
    elseif opt_status == :DUAL_INFEASIBLE
        return MOI.NO_SOLUTION
    elseif opt_status == :OPTIMAL
        return MOI.FEASIBLE_POINT
    else
        error("Unexpected solver status")
        return MOI.INVALID_MODEL
    end
end

# DualStatus
function MOI.get(optimizer::Optimizer, a::MOI.DualStatus)
    if a.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    optimizer.has_result || return MOI.NO_SOLUTION

    opt_status = optimizer.result.status
    if opt_status == :ITERATION_LIMIT
        return MOI.ITERATION_LIMIT
    elseif opt_status ∈ (:TIME_LIMIT, :ITERATION_LIMIT)
        return MOI.NO_SOLUTION
    elseif opt_status == :INFEASIBLE
        return MOI.NO_SOLUTION
    elseif opt_status == :DUAL_INFEASIBLE
        return MOI.NO_SOLUTION
    elseif opt_status == :OPTIMAL
        return MOI.FEASIBLE_POINT
    else
        error("Unexpected solver status")
        return MOI.INVALID_MODEL
    end
end

# ResultCount
MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = optimizer.has_result ? 1 : 0

# ObjectiveValue
function MOI.get(optimizer::Optimizer, a::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, a)
    rawobj = optimizer.result.obj_val + optimizer.objective_constant
    return optimizer.sense == MOI.MAX_SENSE ? -rawobj : rawobj
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{Quadratic}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

# SolveTimeSec
function MOI.get(optimizer::Optimizer, a::MOI.SolveTimeSec)
    !optimizer.has_result && error("Optimizer not called!")
    return optimizer.result.log.setup_time + optimizer.result.log.solve_time
end

# VariablePrimal
function MOI.get(optimizer::Optimizer, a::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, a)
    if optimizer.result.status == :OPTIMAL
        return optimizer.result.x[vi.value]
    end
end

# If your solver returns dual solutions, implement:
# ConstraintDual
function MOI.get(optimizer::Optimizer, a::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(optimizer, a)

    return -optimizer.result.u[constraint_rows(optimizer, ci)] * optimizer.result.ρ
    # if optimizer.result.status == :OPTIMAL
    #     # XXX: negative?
    #     # TODO: Need to multiply by rho
    #     return -optimizer.result.u[constraint_rows(optimizer, ci)] * optimizer.result.ρ
    # else
    #     @warn "Not solved"
    #     # XXX: TODO
    #     # x = optimizer.prim_inf_cert
    #     return nothing
    # end
end

# If your solver accepts primal or dual warm-starts, implement:
# XXX: TODO
# # VariablePrimalStart
# function MOI.set(optimizer::Optimizer, a::MOI.VariablePrimalStart, vi::VI, value)
#     MOI.is_empty(optimizer) && throw(MOI.SetAttributeNotAllowed(a))
#     optimizer.warmstartcache.x[vi.value] = value
# end

# # ConstraintDualStart