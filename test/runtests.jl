using Random, Test
using NDTools
Random.seed!(42)

use_cuda = false
function opt_cu(dat)
    if (use_cuda)
    try
        return cu(dat)
    catch
        return dat
    end
    else
        return dat
    end
end

# include("utils.jl")
include("offset_types.jl")
include("datatype_tools.jl")
include("size_tools.jl")
include("type_tools.jl")
include("iteration_tools.jl")
include("selection_tools.jl")
include("generation_tools.jl")
include("reverse.jl")
include("base_extensions.jl")

use_cuda = true
try
    using CUDA
    if (CUDA.functional())
        @testset "all in CUDA" begin
        include("offset_types.jl")
        include("datatype_tools.jl")
        include("size_tools.jl")
        include("type_tools.jl")
        include("iteration_tools.jl")
        include("selection_tools.jl")
        include("generation_tools.jl")
        include("reverse.jl")
        include("base_extensions.jl")
        end
    end
catch
    @testset "CUDA not available" begin
        @test 1==1
    end
end

return
