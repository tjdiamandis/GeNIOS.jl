using GeNIOS
using Test

using LinearAlgebra, SparseArrays
using Random

include("utils.jl")
@testset "Machine Learning" begin
    include("lasso.jl")
    include("logistic.jl")
end
@testset "Quadratic Programs" begin
    include("portfolio.jl")
end
