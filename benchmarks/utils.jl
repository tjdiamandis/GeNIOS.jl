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

function get_augmented_data(m, n, DATAFILE)
    BLAS.set_num_threads(Sys.CPU_THREADS)
    
    file = CSV.read(DATAFILE, DataFrame)
    M = Matrix{Float64}(file[1:m,:])
    size(M, 1)
    M .= M .- sum(M, dims=1) ./ size(M, 1)
    M .= M ./ std(M, dims=1)
    
    A_non_augmented = @view M[:, 2:end]
    b = @view M[:, 1]
    
    σ = 8
    Ad = zeros(m, n)
    gauss_fourier_features!(Ad, A_non_augmented, σ)
    GC.gc()
    return Ad, b
end

# --- For constrained least squares ---
function construct_problem_constrained_ls(Ad, b)
    m, n = size(Ad)
    P = Ad'*Ad
    q = Ad'*b
    A = I
    l = zeros(n)
    u = ones(n)
    return P, q, A, l, u
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
