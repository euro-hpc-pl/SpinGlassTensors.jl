using Documenter, SpinGlassTensors

_pages = [
    "Introduction" => "index.md",
    "API Reference" => "api.md"
]
# ============================

makedocs(
    sitename="SpinGlassTensors",
    modules = [SpinGlassTensors],
    pages = _pages
    )