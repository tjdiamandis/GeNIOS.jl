module TestGeNIOS

import GeNIOS
using Test

import MathOptInterface as MOI

const OPTIMIZER = MOI.instantiate(
    MOI.OptimizerWithAttributes(GeNIOS.Optimizer, MOI.Silent() => true),
)

const BRIDGED = MOI.instantiate(
    MOI.OptimizerWithAttributes(GeNIOS.Optimizer, MOI.Silent() => true),
    with_bridge_type = Float64,
)

# See the docstring of MOI.Test.Config for other arguments.
const CONFIG = MOI.Test.Config(
    # Modify tolerances as necessary.
    atol = 1e-2,
    rtol = 1e-2,
    # Use MOI.LOCALLY_SOLVED for local solvers.
    optimal_status = MOI.OPTIMAL,
    # Pass attributes or MOI functions to `exclude` to skip tests that
    # rely on this functionality.
    exclude = Any[
        MOI.ConstraintBasisStatus,
        MOI.VariableBasisStatus,
        MOI.ConstraintName,
        MOI.VariableName,
        MOI.ObjectiveBound,
        MOI.DualObjectiveValue,
        MOI.delete
    ],
)

"""
    runtests()

This function runs all functions in the this Module starting with `test_`.
"""
function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

"""
    test_runtests()

This function runs all the tests in MathOptInterface.Test.

Pass arguments to `exclude` to skip tests for functionality that is not
implemented or that your solver doesn't support.
"""
function test_runtests()
    MOI.Test.runtests(
        BRIDGED,
        CONFIG,
        exclude = [
            "test_model_UpperBoundAlreadySet",
            "test_model_LowerBoundAlreadySet",

            # "test_objective_ObjectiveFunction_blank",
            # "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # "test_model_ModelFilter_AbstractConstraintAttribute",
            
            # FIXME
            # See https://github.com/jump-dev/MathOptInterface.jl/issues/1773
            "test_infeasible_",
            # "test_infeasible_MAX_SENSE",
            # "test_infeasible_MAX_SENSE_offset",
            # "test_infeasible_MIN_SENSE",
            # "test_infeasible_MIN_SENSE_offset",
            # "test_infeasible_affine_MAX_SENSE",
            # "test_infeasible_affine_MAX_SENSE_offset",
            # "test_infeasible_affine_MIN_SENSE",
            # "test_infeasible_affine_MIN_SENSE_offset",
            
            # Problem is a nonconvex QP
            "test_quadratic_nonconvex",
            
            # FIXME
            # See https://github.com/jump-dev/MathOptInterface.jl/issues/1759
            "test_unbounded",
            # "test_unbounded_MAX_SENSE",
            # "test_unbounded_MAX_SENSE_offset",
            # "test_unbounded_MIN_SENSE",
            # "test_unbounded_MIN_SENSE_offset",

            "test_solve_DualStatus",
            # FIXME
            "test_model_copy_to_UnsupportedAttribute"
        ], 
        # This argument is useful to prevent tests from failing on future
        # releases of MOI that add new tests. Don't let this number get too far
        # behind the current MOI release though. You should periodically check
        # for new tests to fix bugs and implement new features.
        # exclude_tests_after = v"1.18.0",
    )
    return
end

"""
    test_SolverName()

You can also write new tests for solver-specific functionality. Write each new
test as a function with a name beginning with `test_`.
"""
function test_SolverName()
    @test MOI.get(GeNIOS.Optimizer(), MOI.SolverName()) == "GeNIOS"
    return
end

end # module TestGeNIOS

# This line at tne end of the file runs all the tests!
TestGeNIOS.runtests()