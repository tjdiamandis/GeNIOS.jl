# Some utility functions
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

function add_to_plot!(plt, x, y, label, color; style=:solid, lw=3)
    start = findfirst(y .> 0)
    inds = start:length(x)
    plot!(plt, x[inds], y[inds],
        label=label,
        lw=lw,
        linecolor=color,
        linestyle=style
    )
end

function print_timing_table(names, logs)
    function print_row(row_name, row_field::Symbol, names, logs; func=x->x, unit="s", int_val=false)
        ret = row_name
        for (name, log) in zip(names, logs)
            val = func(getfield(log, row_field))
            if int_val
                ret *= " & " * @sprintf("%4d", val)
            else
                ret *= " & " * @sprintf("%.3f", val) * unit
            end
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
    print_row("\\qquad preconditioner time", :precond_time, names, logs)
    print_row("solve time", :solve_time, names, logs; func=mean)
    print_row("\\qquad number of iterations", :dual_gap, names, logs; func=length, int_val=true)
    print_row("\\qquad avg. linear system time", :linsys_time, names, logs; func=x->mean(1000x), unit="ms")
    print_row("\\qquad avg. prox time", :prox_time, names, logs; func=x->mean(1000x), unit="ms")

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