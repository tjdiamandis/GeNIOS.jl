module GeNIOS

using Random
using LinearAlgebra, SparseArrays
using Printf
using Krylov: CgSolver, cg!, issolved, warm_start!
using RandomizedPreconditioners
using StaticArrays
using LogExpFunctions: log1pexp, logistic
using ForwardDiff:derivative
using Requires: @require
# using LinearMaps
using PrecompileTools

const RP = RandomizedPreconditioners

include("linsys.jl")
include("cones.jl")
include("constraints.jl")
include("types.jl")
include("utils.jl")
include("admm.jl")
include("MOI_wrapper.jl")
include("utils-test.jl")

function __init__()
    @require Optim="429524aa-4258-5aef-a3af-852621145aeb" include("lbfgs.jl")
end

export Solver, MLSolver, QPSolver, GenericSolver
export HessianOperator, SolverOptions
export solve!

include("precompile.jl")


end
