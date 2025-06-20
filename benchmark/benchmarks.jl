using PkgBenchmark, BenchmarkTools
using Random, LinearAlgebra, SparseArrays
using GeNIOS

SUITE = BenchmarkGroup()
for file in readdir(@__DIR__)
    if startswith(file, "bench_") && endswith(file, ".jl")
        SUITE[file[length("bench_") + 1:end - length(".jl")]] =
            include(file)
    end
end