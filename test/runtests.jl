using GeNIOS
using Test

using LinearAlgebra, SparseArrays
using Random

include("utils.jl")
@testset "Lasso" begin
    include("lasso.jl")
end
@testset "Quadratic Programs" begin
    include("portfolio.jl")
end
