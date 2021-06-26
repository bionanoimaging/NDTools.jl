using NDTools, Documenter 

 # set seed fixed for documentation
DocMeta.setdocmeta!(NDTools, :DocTestSetup, :(using NDTools); recursive=true)
makedocs(modules = [NDTools], 
         sitename = "NDTools.jl", 
         pages = Any[
            "NDTools.jl" => "index.md",
            "Utility functions" => "util.md",
         ]
        )

deploydocs(repo = "github.com/RainerHeintzmann/NDTools.jl.git", devbranch="main")
