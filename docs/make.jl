using GeNIOS
using Documenter
using Literate

# For Plots.jl
# https://discourse.julialang.org/t/plotting-errors-when-building-documentation-using-plots-jl-and-documenter-jl/67849
ENV["GKSwstype"]="100"

EXCLUDED_EXAMPLES = []

# utility function from https://github.com/JuliaOpt/Convex.jl/blob/master/docs/make.jl
fix_math_md(content) = replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```")

# utility functions from https://github.com/oxfordcontrol/COSMO.jl/blob/master/docs/make.jl
fix_suffix(filename) = replace(filename, ".jl" => ".md")
function postprocess(content)
      """
      The source files for all examples can be found in [/examples](https://github.com/tjdiamandis/GeNIOS.jl/tree/main/examples).
      """ * content
end

examples_path = joinpath(@__DIR__, "../examples/standard/")
examples = filter(x -> endswith(x, ".jl") && !in(x, EXCLUDED_EXAMPLES), readdir(examples_path))
build_path =  joinpath(@__DIR__, "src", "examples/")

for example in examples
      Literate.markdown(
        examples_path * example, build_path;
        preprocess = fix_math_md,
        postprocess = postprocess,
        flavor = Literate.DocumenterFlavor(),
        credit = true
    )
end
examples_nav = fix_suffix.(joinpath.("examples", examples))

advanced_path = joinpath(@__DIR__, "../examples/advanced/")
advanced = filter(x -> endswith(x, ".jl") && !in(x, EXCLUDED_EXAMPLES), readdir(advanced_path))
build_path_advanced =  joinpath(@__DIR__, "src", "advanced/")

for example in advanced
    Literate.markdown(
        advanced_path * example, build_path_advanced;
        preprocess = fix_math_md,
        postprocess = postprocess,
        flavor = Literate.DocumenterFlavor(),
        credit = true
    )
end
advanced_nav = fix_suffix.(joinpath.("advanced", readdir(joinpath(@__DIR__, "../examples/advanced"))))

makedocs(;
    modules=[GeNIOS],
    authors="Theo Diamandis",
    repo="https://github.com/tjdiamandis/GeNIOS.jl/blob/{commit}{path}#L{line}",
    sitename="GeNIOS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tjdiamandis.github.io/GeNIOS.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => examples_nav,
        "Advanced Usage" => advanced_nav,
        "User Guide" => "guide.md",
        "Solution method" => "method.md",
        "API reference" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/tjdiamandis/GeNIOS.jl",
    devbranch = "main"
)