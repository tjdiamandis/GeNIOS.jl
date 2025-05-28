using Printf, Statistics
using CSV, DataFrames, Statistics, JLD2

# Some utility functions

# Data generation
function gauss_fourier_features!(A_aug, A, σ)
    s = size(A_aug, 2)
    d = size(A, 2)
    W = 1/σ * randn(d, s)
    b = 2π*rand(s)
    mul!(A_aug, A, W)
    A_aug .+= b'
    A_aug .= cos.(A_aug)    
    A_aug .*= sqrt(2 / s) 
    return nothing
end

function get_augmented_data(m, n, datafile)
    BLAS.set_num_threads(Sys.CPU_THREADS)
    
    file = CSV.read(datafile, DataFrame)
    M = Matrix{Float64}(file[1:m,:])
    
    A = @view M[:, 2:end]
    A .= A .- sum(A, dims=1) ./ size(A, 1)
    A .= A ./ std(A, dims=1)
    b = copy(M[:, 1])
    b .= b .- mean(b)
    b .= b ./ std(b)
    
    σ = 8
    Ad = zeros(m, n)
    gauss_fourier_features!(Ad, A, σ)
    GC.gc()
    return Ad, b
end

function load_sparse_data(; file, have_data, dataset_id=1578)
    # news20: 1594
    # real-sim: 1578
    if !have_data
        real_sim = OpenML.load(dataset_id)
    
        b_full = real_sim.class
        A_full = sparse(Tables.matrix(real_sim)[:,1:end-1])
        jldsave(file, A_full, b_full)
        return A_full, b_full
    else
        A_full, b_full = load(file, "A_full", "b_full")
        return A_full, b_full
    end
end


# --- For constrained least squares ---
function construct_problem_constrained_ls(Ad, b)
    m, n = size(Ad)
    P = Ad'*Ad
    q = -Ad'*b
    A = I
    l = zeros(n)
    u = ones(n)
    return P, q, A, l, u
end

function construct_jump_model_elastic_net(A, b, γ, μ)
    m, n = size(A)
    model = Model()
    @variable(model, x[1:n])
    @variable(model, z[1:m])
    @constraint(model, A*x - b .== z)
    
    # Add ℓ1 regularization
    @variable(model, t[1:n])
    @constraint(model, t .>= x)
    @constraint(model, t .>= -x)
    
    # Define objective
    @objective(model, Min, 0.5 * z'*z + 0.5 * μ * x'*x + γ * sum(t))
    return model
end

function softplus(model, t, u)
    z = @variable(model, [1:2], lower_bound = 0.0)
    @constraint(model, sum(z) <= 1.0)
    @constraint(model, [u - t, 1, z[1]] in MOI.ExponentialCone())
    @constraint(model, [-t, 1, z[2]] in MOI.ExponentialCone())
end

function construct_jump_model_logistic(A, b, γ)
    n, p = size(A)
    model = Model()
    @variable(model, x[1:p])
    @variable(model, t[1:n])
    for i in 1:n
        u = (A[i, :]' * x) * b[i]
        softplus(model, t[i], u)
    end
    # Add ℓ1 regularization
    @variable(model, 0.0 <= reg)
    @constraint(model, [reg; x] in MOI.NormOneCone(p + 1))
    # Define objective
    @objective(model, Min, sum(t) + γ * reg)
    return model
end


# Trial running
function run_genios_trial_qp(P, q, A, l, u; options)
    solver = GeNIOS.QPSolver(P, q, A, l, u)
    GC.gc()
    result = solve!(solver; options=options)
    return result
end

# Printing
function print_timing(name, log)
    print("\n$name:")
    @printf("\ntotal time:        %6.4fs", log.solve_time)
    @printf("\n- setup:           %6.4fs", log.setup_time)
    @printf("\n-- pc time:        %6.4fs", log.precond_time)
    @printf("\n- num iter:        %7d", length(log.dual_gap))
    @printf("\n- iter time:       %6.4fs", log.solve_time / length(log.dual_gap))
    @printf("\n-- linsys time:    %6.4fs", mean(log.linsys_time))
    @printf("\n-- prox time:      %6.4fs", mean(log.prox_time))
    return nothing
end

function print_timing_table(names, logs)
    function print_row(row_name, row_field::Symbol, names, logs; func=x->x, unit="s", int_val=false)
        ret = row_name
        for (name, log) in zip(names, logs)
            val = func(getfield(log, row_field))
            ret *= " & " * (int_val ?  @sprintf("%4d", val) : @sprintf("%.3f", val) * unit)
        end
        ret *= "\\\\"
        println(ret)
        return nothing
    end

    n_exp = length(names)
    println("\\begin{tabular}{@{}l" * "r"^n_exp * "@{}}")
    println("\\toprule")
    println("&" * join(names, " & ") * "\\\\")
    println("\\midrule")
    print_row("setup time (total)", :setup_time, names, logs)
    print_row("\\quad preconditioner time", :precond_time, names, logs)
    print_row("solve time", :solve_time, names, logs; func=mean)
    print_row("\\quad number of iterations", :dual_gap, names, logs; func=length, int_val=true, unit="")
    print_row("\\quad total linear system time", :linsys_time, names, logs; func=x->sum(x), unit="s")
    print_row("\\quad avg. linear system time", :linsys_time, names, logs; func=x->mean(1000x), unit="ms")
    print_row("\\quad total prox time", :prox_time, names, logs; func=x->sum(x), unit="s")
    print_row("\\quad avg. prox time", :prox_time, names, logs; func=x->mean(1000x), unit="ms")

    # total time
    ret = "total time"
    for (name, log) in zip(names, logs)
        val = log.setup_time + log.solve_time
        ret *= " & " * @sprintf("%.3f", val) * "s"
    end
    ret *= "\\\\"
    println(ret)
    println("\\bottomrule")
    println("\\end{tabular}")
end

# Plotting
function add_to_plot!(plt, x, y, label, color; style=:solid, lw=3)
    start = findfirst(y .!= 0)
    inds = start:length(x)
    plot!(plt, x[inds], y[inds],
        label=label,
        lw=lw,
        linecolor=color,
        linestyle=style
    )
end
