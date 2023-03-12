using Documenter, SpinGlassTensors

_pages = [
    "Introduction" => "index.md",
    "API Reference" => "api.md"
]
# ============================

format = Documenter.HTML(edit_link = "master",
                         prettyurls = get(ENV, "CI", nothing) == "true",
)

# format = Documenter.LaTeX(platform="none")

makedocs(
    sitename="SpinGlassTensors.jl",
    modules = [SpinGlassTensors],
    pages = _pages,
    format = format
    )