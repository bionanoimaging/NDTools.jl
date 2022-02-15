using NDTools, Documenter 

 # set seed fixed for documentation
DocMeta.setdocmeta!(NDTools, :DocTestSetup, :(using NDTools); recursive=true)
makedocs(modules = [NDTools], 
         sitename = "NDTools.jl", 
         pages = Any[
            "NDTools.jl" => "index.md",
            "Center Methods" => "center.md",
            "Select Regions" => "select_region.md",
            "Dimensionality Functions" => "dims.md",
            "Reverse views" => "reverse_view.md",
         ]
        )

deploydocs(repo = "github.com/RainerHeintzmann/NDTools.jl.git", devbranch="main")
