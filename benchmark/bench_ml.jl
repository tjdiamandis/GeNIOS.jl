module BenchML

using BenchmarkTools
using GeNIOS

const SUITE = BenchmarkGroup()

A, b, xstar, γmax = GeNIOS.generate_lasso_problem(100, 200; rseed=1)
solver = GeNIOS.LassoSolver(0.05*γmax, A, b)

SUITE["lasso"] = @benchmarkable solve!($solver; options=GeNIOS.SolverOptions(verbose=false))


end
BenchML.SUITE