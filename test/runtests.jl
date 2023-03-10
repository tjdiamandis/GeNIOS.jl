using GeNIOS
using Test

using LinearAlgebra, SparseArrays
using Random

@testset "ADMM" begin
    include("admm.jl")
end
