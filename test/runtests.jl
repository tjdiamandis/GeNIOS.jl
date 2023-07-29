using GeNIOS
using Test

using LinearAlgebra, SparseArrays, Random
using Optim

include("utils.jl")

@testset "Machine Learning" begin
    include("lasso.jl")
    include("logistic.jl")
end

@testset "Quadratic Programs" begin
    include("portfolio.jl")
    include("randqp.jl")
    include("infeas.jl")
end

@testset "Conic Intercace" begin
    include("constraints.jl")
end

@testset "MOI" begin
    include("MOI_wrapper.jl")
end
