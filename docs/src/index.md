```@meta
CurrentModule = GeNIOS
```

# GeNIOS.jl
TODO

### Documentation Contents:
```@contents
Pages = ["index.md", "method.md", "guide.md"]
Depth = 1
```
##### Examples:
```@contents
Pages = ["examples/constrained-ls.md"]
Depth = 1
```


## Overview

GeNIOS solves convex programs of the form

```math
\begin{array}{ll}
\text{minimize}     & f(x) + g(z) \\
\text{subject to}   & Mx + z = c,
\end{array}
```
where $x \in \mathbb{R}^n$ and $z \in \mathbb{R}^m$ are the optimization variables.



Many important problems in machine learning, finance, operations research, and control
can be formulated as QPs. Examples include
- Lasso regression
- Portfolio optimization
- Trajectory optimization
- Model predictive control
- And many more...


### Algorithm
GeNIOS follows a similar approach to OSQP [^1], solving the QP using ADMM [^2].
The main difference between GeNIOS and OSQP is the use of a randomized method to solve the linear system
at each iteration, which provides a significant speedup in practice (often over XX%!).
In addition, GeNIOS incorporates several novel heuristics.
Check out [Algorithm](@ref) for additional details.

## Getting Started
The JuMP interface is the easiest way to use GeNIOS.
A simple example is below.
```julia
using JuMP, GeNIOS
# TODO:
```


However, the native interface can be called directly by specifying the problem data:
```julia
#TODO
```

Check out [User Guide](@ref) for keyword arguments and performance tips.

## What's with the randomness?
The primary bottleneck of the ADMM algorithm is a linear system solve at each iteration.
GeNIOS uses the conjugate gradient method with a randomized Nyström preconditioner [^3]
to solve this system. 
This randomized approach often provides a significant speedup.

TODO: picture

## References
[^1]: Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). [OSQP: An operator splitting solver for quadratic programs](https://osqp.org). 

[^2]: Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). [Distributed optimization and statistical learning via the alternating direction method of multipliers](https://stanford.edu/~boyd/admm.html).

[^3]: Frangella, Z., Tropp, J. A., & Udell, M. (2021). [Randomized Nyström Preconditioning](https://arxiv.org/pdf/2110.02820.pdf).