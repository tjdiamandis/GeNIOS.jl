module GeNIOS

using Random
using LinearAlgebra, SparseArrays
using Printf
using Krylov: CgSolver, cg!, issolved, warm_start!
using RandomizedPreconditioners
using StaticArrays
using LogExpFunctions: log1pexp, logistic

const RP = RandomizedPreconditioners

include("linsys.jl")
include("cones.jl")
include("types.jl")
include("utils.jl")
include("admm.jl")

export Solver, solve!
export HessianOperator


end
