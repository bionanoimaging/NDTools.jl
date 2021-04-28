module NDTools
using Base.Iterators


const IterType = Union{NTuple{N,Tuple} where N, Vector, Matrix, Base.Iterators.Repeated}

# Functions from ROIViews:

"""
    expand_add(t1,t2)  # adds t1 to t2 as a tuple and returns t2[n] for n > length(t1)
adds the elements of the tuple t1 to t2 for components in t1 which exist for the rest just t2 is used.
```jldoctest
julia> expand_add((1,2,3),(4,5,6,7,8,9))
(5, 7, 9, 7, 8, 9)
```
"""
function expand_add(t1,t2)  # adds t1 to t2 as a tuple and returns t2[n] for n > length(t1)
    ((t+w for (t,w) in zip(t1,t2))..., (w for w in t2[length(t1)+1:end])...)
    # ((t+w for (t,w) in zip(t1,t2))...,t2[length(t1)+1:end]...)
end

"""
    expand_size(sz,sz2)
expands a size tuple sz with the sizes as given in the tuple sz2 for positions which do not exist in sz
```jldoctest
julia> sz = expand_size((1,2,3),(4,5,6,7,8,9))
Base.Generator{UnitRange{Int64}, var"#7#8"{Tuple{Int64, Int64, Int64}, NTuple{6, Int64}, Int64}}(var"#7#8"{Tuple{Int64, Int64, Int64}, NTuple{6, Int64}, Int64}((1, 2, 3), (4, 5, 6, 7, 8, 9), 3), 1:6)

julia> collect(sz)
6-element Vector{Int64}:
 1
 2
 3
 7
 8
 9
```
"""
function expand_size(sz,sz2)
    dims1 = length(sz)
    dims2 = length(sz2)
    ((d<=dims1) ? sz[d] : sz2[d] for d in 1:dims2)
end

# Functions from IndexFunArrays:

"""
    optional_posZ(x, offset)
returns a z position `posZ = x[3]-offset[3]` if the tuple is long enough and `posZ=1` otherwise.
This is useful for 3d routines, which should also work for 2d data.
```jldoctest
julia> optional_posZ((5,5),(2,3))
1

julia> optional_posZ((5,5,5),(3,2,1))
4
```
"""
optional_posZ(x::NTuple{1,T}, offset::NTuple{1,T}) where {T,N} = 1
optional_posZ(x::NTuple{2,T}, offset::NTuple{2,T}) where {T,N} = 1
optional_posZ(x::NTuple{N,T}, offset::NTuple{N,T}) where {T,N} = x[3]-offset[3]

"""
    curry(f, x) = (xs...) -> f(x, xs...)   # just a shorthand to remove x
allows to remove the fix the first argument x in a function f with any number of arguments
```jldoctest
julia> g = curry(+,10.0)
#9 (generic function with 1 method)

julia> g(3)
13.0
```
"""
curry(f, x) = (xs...) -> f(x, xs...)   # just a shorthand to remove x

## These functions ensure that also numbers can be iterated and zipped
function cast_iter(vals::Matrix)
    Tuple(Tuple(vals[:,n]) for n in 1:size(vals,2))
    # Tuple(vals[:,axes(vals, 2)])
end

function cast_iter(vals::Vector{<:Number})
    Tuple(vals)   # makes the matrix iterable, interpreting it as a series of vectors
end

function cast_iter(vals::IterType)
    vals
end

function cast_iter(vals)
    repeated(vals) # during the zip operation this type always yields the same value
end

function cast_number_iter(vals::Vector{<:Number})
    Tuple(vals)   # makes the matrix iterable, interpreting it as a series of vectors
end

function cast_number_iter(vals::NTuple{N,<:Number} where N)
    vals
end

function cast_number_iter(vals::Number)
    repeated(vals) # during the zip operation this type always yields the same value
end


function optional_mat_to_iter(vals)  # only for matrices
    vals
end

function optional_mat_to_iter(vals::Matrix)
    cast_iter(vals)
end

mat_to_tvec(v) = [Tuple(v[:,n]) for n in 1:size(v,2)] # converts a 2d matrix to a Vector of Tuples

function apply_dims(scale, dims, N)  # replaces scale entries not in dims with zeros
    ntuple(i -> i ∈ dims ? scale[i] : zero(scale[1]), N)
end

function apply_dims(scales::IterType, dims, N)  # replaces scale entries not in dims with zeros
    Tuple(ntuple(i -> i ∈ dims ? scale[i] : zero(scale[1]), N)  for scale in  scales)
end

# Functions for IndexFunArrays.utils

"""
    single_dim_size(dim::Int,dim_size::Int)

Returns a tuple (length `dim`) of singleton sizes except at the final position `dim`, which contains `dim_size`

# Example
```jldoctest
julia> IndexFunArrays.single_dim_size(4, 3)
(1, 1, 1, 3)

julia> IndexFunArrays.single_dim_size(4, 5)
(1, 1, 1, 5)

julia> IndexFunArrays.single_dim_size(2, 5)
(1, 5)
```
"""
function single_dim_size(dim::Int,dim_size::Int)
    Base.setindex(Tuple(ones(Int, dim)),dim_size,dim)
end


"""
    selectsizes(x::AbstractArray, dim; keep_dims=true)

Additional size method to access the size at several dimensions
in one call.
`keep_dims` allows to return the other dimensions as singletons.

# Examples
```jldoctest
julia> x = ones((2,4,6,8, 10));

julia> selectsizes(x, (2,3))
(1, 4, 6, 1, 1)

julia> selectsizes(x, 5)
(1, 1, 1, 1, 10)

julia> selectsizes(x, (5,))
(1, 1, 1, 1, 10)

julia> selectsizes(x, (2,3,4), keep_dims=false)
(4, 6, 8)

julia> selectsizes(x, (4,3,2), keep_dims=false)
(8, 6, 4)
```

"""
function selectsizes(x::AbstractArray{T},dim::NTuple{N,Int};
                    keep_dims=true) where{T,N}
    if ~keep_dims
        return map(n->size(x,n),dim)
    end
    sz = ones(Int, ndims(x))
    for n in dim
        sz[n] = size(x,n) 
    end
    return Tuple(sz)
end 

function selectsizes(x::AbstractArray, dim::Integer; keep_dims=true)
    selectsizes(x, Tuple(dim), keep_dims=keep_dims)
end


# These are the type promotion rules, taken from float.jl but written in terms of types
# see also 
promote_type()
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

# Functions from scaling_offset_types in IndexFunArrays

# define types to specify where the center point is
abstract type Ctr end  # Center of the array
struct CtrCorner <: Ctr end  # corner voxel is zero
struct CtrFFT <: Ctr end # corresponding to FFTs
struct CtrRFFT <: Ctr end # corresponding to RFFTs
struct CtrFT <: Ctr end # corresponding to FTs  (meaning shifted FFTs)
struct CtrRFT <: Ctr end # corresponding to RFTs
struct CtrMid <: Ctr end # middle of the array
struct CtrEnd <: Ctr end # other corner voxel is zero

"""
    Ctr

Abstract type to specify the reference position
from which several other types subtype.

# Possible subtypes
* `CtrCorner`: Set the reference pixel in the corner
* `CtrFFT`: Set the reference pixel to the FFT center.
* `CtrFT`: Set the reference pixel to the FT center. FT means that the zero frequency is at the FFT convention center (`size ÷ 2 + 1`).
* `CtrRFFT`: Set the reference pixel to the RFFT center. Same as `CtrFFT` but the first dimension has center at 1. 
* `CtrRFT`: Set the reference pixel to the RFT center. FT means that the zero frequency is at the FFT convention center (`size ÷ 2 + 1`). 
            Same as `CtrFT` but the first dimension has center at 1.
* `CtrMid`: Set the reference pixel to real mid. For uneven arrays it is the center pixel, for even arrays it is the centered around a half pixel.
* `CtrEnd` Set the reference to the end corner (last pixel)
"""
Ctr

# In the equations below, it is important to not use 1.0 or 0.0 as these would cause the result to be of type FLoat64
get_offset(size, ::Type{CtrCorner}) = size.*0 .+ 1
get_offset(size, ::Type{CtrFT}) = size.÷2 .+ 1
get_offset(size, ::Type{CtrRFT}) = Base.setindex(size.÷2,0,1) .+ 1
get_offset(size, ::Type{CtrFFT}) = size.*0 .+ 1
get_offset(size, ::Type{CtrRFFT}) = size.*0 .+ 1
get_offset(size, ::Type{CtrMid}) = (size.+1)./2
get_offset(size, ::Type{CtrEnd}) = size.+0
get_offset(size, t::Number) = ntuple(i -> t, length(size))
get_offset(dummy, t::NTuple) = t
get_offset(dummy, t::IterType) = t
get_offset(dummy, t::Matrix) = Tuple(Tuple(t[:,n]) for n in 1:size(t,2))  # converts the matrix to an iterable collection for convenience

abstract type Sca end # scaling of the array
struct ScaUnit <: Sca end # pixel distance is one
struct ScaNorm <: Sca end # total size along each dimension normalized to 1.0
struct ScaMid <: Sca end # such that the border pixel always reaches 1.0
struct ScaFT <: Sca end # reciprocal Fourier coordinates
struct ScaRFT <: Sca end # reciprocal Fourier coordinates for rFTs. 
struct ScaFTEdge <: Sca end # such that the edge of the Fourier space is 1.0
struct ScaRFTEdge <: Sca end # such that the edge of the Fourier space is 1.0

"""
    Sca 

Abstract type to indicate a scaling from which several other types subtype.

# Possible subtypes
* `ScaUnit`: No scaling of the indices 
* `ScaNorm`: Total length along each dimension is normalized to 1
* `ScaMid`: Reaches 1.0 at the borders, if used in combination with `CtrMid`. Useful for keeping real-space symmetry.
* `ScaFT`: Reciprocal Fourier coordinates compared to Nyquist sampling
* `ScaFTEdge`: Such that the edge (in FFT sense) of the pixel is 1.0
"""
Sca

get_scale(sz, ::Type{ScaUnit}) = ntuple(_ -> one(Int), length(sz))
get_scale(sz, ::Type{ScaNorm}) = 1 ./ (max.(sz .- 1, 1)) 
get_scale(sz, ::Type{ScaMid}) = 1 ./ (max.((sz .- 1)./2, 1)) # reaches 1.0 at the outermost pixel. 
get_scale(sz, ::Type{ScaFT}) = 0.5 ./ (max.(sz ./ 2, 1))
# get_scale(size, ::Type{ScaRFT}) = 0.5 ./ (max.(Base.setindex(size./ 2,size[1]-1,1), 1))  # These scales are wrong! They need the information on the real-space size!
get_scale(sz, ::Type{ScaFTEdge}) = 1 ./ (max.(sz ./ 2, 1))  
# get_scale(size, ::Type{ScaRFTEdge}) = 1 ./ (max.(Base.setindex(size./ 2,size[1]-1,1), 1))
get_scale(sz, t::Number) = ntuple(i -> t, length(sz)) 
get_scale(dummy, t::NTuple) = t 
get_scale(dummy, t::IterType) = t
get_scale(dummy, t::Matrix) = Tuple(Tuple(t[:,n]) for n in 1:size(t,2))  # converts the matrix to an iterable collection for convenience

function apply_tuple_list(f, t1,t2)  # applies a two-argument function to tubles and iterables of tuples
    return f(t1,t2)
end

function apply_tuple_list(f, t1,t2::IterType)
    return Tuple([f(t1,a2) for a2 in t2])
end

function apply_tuple_list(f, t1::IterType,t2)
    res= Tuple([f(a1,t2) for a1 in t1])
    return res
end

function apply_tuple_list(f, t1::IterType,t2::IterType)
    return Tuple([f(a[1],a[2]) for a in zip(t1,t2)])
end


## Functions from FourierTools utils:

"""
    selectsizes(x, dism; keep_dims=true)

Select the sizes of `x` for all `dims`
If `keep_dims=true` the non-selected dimensions are
returned as 1.

# Examples
```jldoctest
julia> FourierTools.selectsizes(randn((4,3,2)), (2,3))
(1, 3, 2)

julia> FourierTools.selectsizes(randn((4,3,2)), (2,3), keep_dims=false)
(3, 2)
```

"""
function selectsizes(x::AbstractArray{T},dims::NTuple{N,Int};
                    keep_dims=true) where{T,N}
    if ~keep_dims
        return map(n->size(x,n),dims)
    end
    sz = ones(Int, ndims(x))
    for n in dims
        sz[n] = size(x,n) 
    end
    return Tuple(sz)
end 



"""
    slice(arr, dim, index)
Return a `N` dimensional slice (where one dimensions has size 1) of the N-dimensional `arr` at the index position
`index` in the `dim` dimension of the array.
It holds `size(out)[dim] == 1`.
# Examples
```jldoctest
julia> x = [1 2 3; 4 5 6; 7 8 9]
3×3 Matrix{Int64}:
 1  2  3
 4  5  6
 7  8  9

julia> FourierTools.slice(x, 1, 1)
1×3 view(::Matrix{Int64}, 1:1, 1:3) with eltype Int64:
 1  2  3
```
"""
function slice(arr::AbstractArray{T, N}, dim::Integer, index::Integer) where {T, N}
    inds = slice_indices(axes(arr), dim, index)
    return @view arr[inds...]
end

"""
    slice_indices(a, dim, index)

`a` should be the axes obtained by `axes(arr)` of an array.
`dim` is the dimension to be selected and `index` the index of it.

# Examples
```jldoctest
julia> FourierTools.slice_indices((1:10, 1:20, 1:12, 1:33), 1, 3)
(3:3, 1:20, 1:12, 1:33)
```
"""
function slice_indices(a::NTuple{N, T}, dim::Integer, index::Integer) where {T, N}
    inds = ntuple(i -> i == dim ? (a[i][index]:a[i][index])
                                : (first(a[i]):last(a[i])), 
                  Val(N))
    return inds
end

"""
    expanddims(x, ::Val{N})
    expanddims(x, N::Number)

expands the dimensions of an array to a given number of dimensions.

Try to prefer the `Val` version because this is type-stable.
`Val(N)` encapsulates the number in a type from which the compiler
can then infer the return type.

# Examples
The result is a 5D array with singleton dimensions at the end
```jldoctest
julia> expanddims(ones((1,2,3)), Val(5))
1×2×3×1×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 1.0  1.0

[:, :, 2, 1, 1] =
 1.0  1.0

[:, :, 3, 1, 1] =
 1.0  1.0

julia> expanddims(ones((1,2,3)), 5)
1×2×3×1×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 1.0  1.0

[:, :, 2, 1, 1] =
 1.0  1.0

[:, :, 3, 1, 1] =
 1.0  1.0
```
"""
function expanddims(x, N::Number)
    return reshape(x, (size(x)..., ntuple(x -> 1, (N - ndims(x)))...))
end

function expanddims(x, ::Val{N}) where N
    return reshape(x, (size(x)..., ntuple(x -> 1, (N - ndims(x)))...))
end


end # module
