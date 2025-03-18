module NDTools
using Core: add_int
using Base.Iterators, LinearAlgebra
using Statistics
using MutableShiftedArrays

include("type_tools.jl")
include("offset_types.jl")
include("datatype_tools.jl")
include("size_tools.jl")
include("iteration_tools.jl")
include("selection_tools.jl")
include("generation_tools.jl")
include("reverse.jl")
include("base_extensions.jl")

end # module
