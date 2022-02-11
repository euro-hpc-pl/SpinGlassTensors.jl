using Documenter, SpinGlassTensors
makedocs(
    modules=[SpinGlassTensors],
    sitename="SpinGlassTensors.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)
deploydocs(
    repo = "github.com/euro-hpc-pl/SpinGlassTensors.jl.git",
)
