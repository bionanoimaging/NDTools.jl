export real_arr_type, complex_arr_type, similar_arr_type

"""
    real_arr_type(::Type{TA}) where {TA<:AbstractArray}

returns the same array type but using `(real(eltype()))` as the element type
# Arguments
+ `TA`:     The array type to convert to an eltype of `real(eltype(TA))`
+ `dims`:   The number of dimensions of the returned array type

# Example
```jldoctest
julia> real_arr_type(Array{ComplexF64})
Vector{Float64} (alias for Array{Float64, 1})

julia> real_arr_type(Array{ComplexF64,3})
Array{Float64, 3}

julia> real_arr_type(Array{ComplexF64,3}, dims=4)
Array{Float64, 4}
```
"""
function real_arr_type(::Type{TA}; dims=N) where {T,N, TA<:AbstractArray{T,N}}
    similar_arr_type(TA, dtype=real(eltype(TA)), dims=dims)
end

function real_arr_type(::Type{TA}; dims=1) where {TA<:AbstractArray}
    similar_arr_type(TA, dtype=real(eltype(TA)), dims=dims)
end

"""
    complex_arr_type(::Type{TA}) where {TA<:AbstractArray}

returns the same array type but using `(complex(eltype()))` as the element type

# Arguments
+ `TA`:     The array type to convert to an eltype of `complex(eltype(TA))`
+ `dims`:   The number of dimensions of the returned array type

# Example
```jldoctest
julia> complex_arr_type(Array{Float32})
Vector{ComplexF32} (alias for Array{Complex{Float32}, 1})

julia> complex_arr_type(Array{Float32,3})
Array{ComplexF32, 3}

julia> complex_arr_type(Array{Float32,3},dims=1)
Vector{ComplexF32} (alias for Array{Complex{Float32}, 1})
```
"""
function complex_arr_type(::Type{TA}; dims=N) where {T,N, TA<:AbstractArray{T,N}}
    similar_arr_type(TA, dtype=complex(eltype(TA)), dims=dims)
end

function complex_arr_type(::Type{TA}; dims=1) where {TA<:AbstractArray}
    similar_arr_type(TA, dtype=complex(eltype(TA)), dims=dims)
end
 

"""
    similar_arr_type(::Type{TA}; dims=N, dtype=T) where {TA<:AbstractArray}

returns a similar array type but using as TA, but eltype and ndims can be changed.

# Arguments
+ `TA`:     The array type to convert to an eltype of `complex(eltype(TA))`
+ `dims`:   The number of dimensions of the returned array type. Please specify this as a ::Val type to be type-stable. Default is Val(1).
+ `dtype`:  The `eltype()` of the returned array type.

# Example

```jldoctest
julia> similar_arr_type(Array{ComplexF64})
Vector{ComplexF64} (alias for Array{Complex{Float64}, 1})

julia> similar_arr_type(Array{ComplexF64,3})
Array{ComplexF64, 3}

julia> similar_arr_type(Array{ComplexF64,3}, dims=2, dtype=Int)
Matrix{Int64} (alias for Array{Int64, 2})
```
"""
function similar_arr_type(::Type{TA}; dims=N, dtype=T) where {T, N, TA<:AbstractArray{T,N}}
    typeof(similar(TA(undef, ntuple(x->0, N)), dtype, ntuple(x->0, dims)))
end

function similar_arr_type(::Type{TA}; dims=N, dtype=T) where {T, N, P, I, L, TA<:SubArray{T,N,P,I,L}}
    similar_arr_type(P, dims=dims, dtype=dtype)
end

function similar_arr_type(::Type{TA}; dims=N, dtype=T) where {T, N, P, MI, TA<:Base.ReshapedArray{T,N,P,MI}}
    similar_arr_type(P, dims=dims, dtype=dtype)
end

# note that T refers to the new type (if not explicitely specified) and therefore replaces the eltype of the array as defined by P
function similar_arr_type(::Type{TA}; dims=N, dtype=T) where {T, N, O, P, B, TA<:Base.ReinterpretArray{T,N,O,P,B}}
    similar_arr_type(P, dims=dims, dtype=dtype)
end

function similar_arr_type(::Type{TA}; dims=Val(1), dtype=eltype(TA)) where {TA<:AbstractArray}
    typeof(similar(TA(undef), dtype, ntuple(x->0, dims)))
end

