#=
# Constrained Least Squares
This example setsup a constrained least squares problem using the quadratic program interface
of GeNIOS.jl. It is from the [OSQP docs](https://osqp.org/docs/examples/least-squares.html).

Specifically, we want to solve the problem
$$
\begin{array}{ll}
\text{minimize}     & \|Ax - b\|_2^2 \\
\text{subject to}   & 0 \leq x \leq 1.
\end{array}
$$
=#

using GeNIOS
using Random, LinearAlgebra, SparseArrays
using JuMP

#=
## Generating the problem data
=#
Random.seed!(1)
m, n = 30, 20
Ad = sprandn(m, n, 0.7)
b = randn(m);

#=
## Using JuMP
We can model this problem using JuMP and then pass it to GeNIOS.
=#
model = JuMP.Model(GeNIOS.Optimizer)
@variable(model, x[1:n])
@objective(model, Min, sum((Ad*x - b).^2))
@constraint(model, 0 .<= x)
@constraint(model, x .<= 1)
optimize!(model)

## Show optimal objective value
println("Optimal value: $(round(objective_value(model), digits=4))")

#=
## QPSolver interface
To make the formulations match, we will introdudce a new variable $y = Ax - b$.
(GeNIOS does not support constants in the objective function.) 
The problem becomes

$$
\begin{array}{ll}
\text{minimize}     & y^Ty \\
\text{subject to}   & Ax - y = b \\
                    & 0 \leq x \leq 1.
\end{array}
$$

In OSQP form, this problem is

$$
\begin{array}{ll}
\text{minimize}     & y^Ty \\
\text{subject to}   & Ax - y = z_1 \\
                    & x = z_2 \\
                    & b \leq z_1 \leq b \\
                    & 0 \leq z_2 \leq 1
\end{array}
$$
or more explicitly
$$
\begin{array}{ll}
\text{minimize}     & \begin{bmatrix} x \\ y \end{bmatrix}^T \begin{bmatrix} 0 & \\  & I \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} \\
\text{subject to}   & \begin{bmatrix} A & -I \\ I & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = z
                    & \begin{bmatrix} b \\ 0 \end{bmatrix} \leq z \leq \begin{bmatrix} b \\ 1 \end{bmatrix}.
\end{array}
$$
=#

P = blockdiag(spzeros(n, n), 2*sparse(I, m, m))
q = spzeros(n + m)
M = [
    Ad                  -sparse(I, m, m);
    sparse(I, n, n)     spzeros(n, m)
]
l = [b; spzeros(n)]
u = [b; ones(n)];

#=
## Building and solving the problem
Now we setup the solver and solve the problem
=#
solver = GeNIOS.QPSolver(P, q, M, l, u)
res = solve!(solver)

#=
## Examining the solution
We can check the solution and its optimality
=#
xstar = solver.zk[m+1:end]
ls_residual =  1/√(m) * norm(Ad*xstar - b)
feas = all(xstar .>= 0) && all(xstar .<= 1)
println("Is feasible? $feas")
println("Least Squres RMSE = $(round(ls_residual, digits=8))")
println("Primal residual: $(round(solver.rp_norm, digits=8))")
println("Dual residual: $(round(solver.rd_norm, digits=8))")
println("Optimal value: $(round(solver.obj_val, digits=4))")