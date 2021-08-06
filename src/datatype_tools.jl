
## datatype_tools.jl
export get_complex_datatype

const IterType = Union{NTuple{N,Tuple} where N, Vector, Matrix, Base.Iterators.Repeated}

# These are the type promotion rules, taken from float.jl but written in terms of types
# see also promote_type()
default_type(::Type{Bool}, def_T)    = def_T
default_type(::Type{Int8}, def_T)    = def_T
default_type(::Type{Int16}, def_T)   = def_T
default_type(::Type{Int32}, def_T)   = def_T
default_type(::Type{Int64}, def_T)   = def_T # LOSSY
default_type(::Type{Int128}, def_T)  = def_T # LOSSY
default_type(::Type{UInt8}, def_T)   = def_T
default_type(::Type{UInt16}, def_T)  = def_T
default_type(::Type{UInt32}, def_T)  = def_T
default_type(::Type{UInt64}, def_T)  = def_T # LOSSY
default_type(::Type{UInt128}, def_T) = def_T # LOSSY
default_type(::Type{T}, def_T) where{T} = T # all other types remain to be the same


"""
    get_complex_datatype(x)

returns the complex-valued datatyp which encompasses the eltype

Examples:
```jdoctest
julia> get_complex_datatype([1f0,2f0,3f0])
ComplexF32 (alias for Complex{Float32})

julia> get_complex_datatype(22.2)
ComplexF64 (alias for Complex{Float64})
```
"""
get_complex_datatype(x :: Number) = Complex{typeof(x)}
get_complex_datatype(x :: Complex ) = typeof(x)
get_complex_datatype(x :: AbstractArray) = get_complex_datatype(eltype(x)(0))


