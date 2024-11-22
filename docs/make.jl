using HuggingFaceTokenizers
using Documenter

DocMeta.setdocmeta!(HuggingFaceTokenizers, :DocTestSetup, :(using HuggingFaceTokenizers); recursive=true)

makedocs(;
    modules=[HuggingFaceTokenizers],
    authors="AntonOresten <anton.oresten42@gmail.com> and contributors",
    sitename="HuggingFaceTokenizers.jl",
    format=Documenter.HTML(;
        canonical="https://AntonOresten.github.io/HuggingFaceTokenizers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/AntonOresten/HuggingFaceTokenizers.jl",
    devbranch="main",
)
