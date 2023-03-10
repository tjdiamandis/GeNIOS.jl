# --- Solver printing utils ---
function print_header(format, data)
    @printf(
        "\n──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
end


function print_footer()
    @printf(
        "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
    )
end


function print_iter_func(format, data)
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
end
