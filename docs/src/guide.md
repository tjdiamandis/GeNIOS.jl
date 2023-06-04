# User Guide

To get started, check out the **Examples**.
The **Advanced Examples** demonstrate ways to improve performance for a
variety of different problem types.

## Solver parameters
The important ADMM parameters are saved as part of the `Solver` object:
- `α`
    - Over-relaxation parameter. Note that `relax` option (in `SolverOptions`, see [Algorithm parameters](@ref)) must be `true`, which is the default to use over-relaxation.
    - Default value: `1.6`
- `ρ`
    - ADMM dual update parameter
    - Default value: `1.0`

## Algorithm parameters
The `GeNIOS` algorithm has the following tunable parameters, which are tunable
either through the JuMP interface or via the [`SolverOptions`](@ref) object,
passed to `solve!`:

- `relax::Bool` 
    - Toggles if over-relaxation is used. Empirically, over-relaxation has been
    shown to improve convergence.
    - default value: `true`
- `logging::Bool` 
    - Toggles if the iterates are logged (objective value, iteration time, linear 
    system solve time, primal and dual residuals, dual gap (if applicable))
    - default value: `true`
- `use_dual_gap::Bool` 
    - (`MLSolver` only) Toggle if dual gap is used as the convergence criterion
    - default value: `false`
- `dual_gap_tol::Real` 
    - (`MLSolver` only) Dual gap convergence tolerance
    - default value: `1e-4`
- `max_iters::Int` 
    - Maximum number of allowed solver iterations
    - default value: `1000`
- `max_time_sec::Real` 
    - Maximum allowed solve time
    - default value: `1200.0`
- `print_iter::Int` 
    - Printing frequency. Every `print_iter` iterations, the solver prints the
    objective value, convergence criteria, `ρ` value, and time. The `MLSolver`
    also prints the root mean squared error (square root of the average per-sample
    loss)
    - default value: `25`
- `rho_update_iter::Int` 
    - Frequency at which the parameter `ρ` is updated
    - default value: `50`
- `sketch_update_iter::Int` 
    - Frequency at which the preconditioner is updated
    - default value: `20`
- `verbose::Bool` 
    - Printing toggle
    - default value: `true`
- `multithreaded::Bool` 
    - Multithreading toggle
    - default value: `false`
- `linsys_max_tol::Real` 
    - Maximum relative error tolerance for the linear system solve. This criterion
    mostly matters for the first iteration; we use a criterion based on the 
    square root of the residuals as soon as they are sufficiently small
    - default value: `1e-1`
- `eps_abs::Real` 
    - Absolute stopping tolerance (applied to residuals)
    - default value: `1e-4`
- `eps_rel::Real` 
    - Relative stopping tolerance (applied to residuals)
    - default value: `1e-4`
- `eps_inf::Real` 
    - Infeasibility detection tolerance
    - default value: `1e-8`
- `norm_type::Real` 
    - Norm to use for convergence criteria
    - default value: `2`

### Nyström PCG Parameters
A subset of the `SolverOptions` parameters deal with the preconditioner directly:

- `precondition::Bool` 
    - Toggles if the linear system solve for the $x$-update is preconditioned
    - default value: `true`
- `update_preconditioner::Bool` 
    - Preconditioner update toggle. If the Hessian is known to be a constant
    (_e.g._, in the lasso problem), this should be set to `false`.
    - default value: `true`
- `init_sketch_size`::Int
    - Initial sketch size used for the preconditioner. Note that the solver
    will use the smaller of this and the side length divided by 10.
    - default value: `50`
- `use_adaptive_sketch::Bool`
    - If true, the solver resketches the matrix and increases the sketch size
    until a pre-defined tolerance is met.
    - default value: `false`
- `adaptive_sketch_tol::Real`
    - If using an adaptive sketch, increases the sketch rank until the estimated
    spectral norm is under this tolerance times $n^2$
    - default value: `eps()`

