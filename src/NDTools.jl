module NDTools
using Core: add_int
using Base.Iterators, PaddedViews, LinearAlgebra, IndexFunArrays, Statistics
export collect_dim, selectdim, select_sizes, expand_add, expand_size, expand_dims, 
       apply_tuple_list, reorient, select_region, select_region!, single_dim_size
export get_complex_datatype, center_position, center_value, pack
export soft_theta, exp_decay, multi_exp_decay, soft_delta, radial_mean, linear_index, Δ_phase
export get_scan_pattern, flatten_trailing_dims
export image_to_arr, moment_proj_normed, idx_to_dim, ϕ_tuple
export assign_to!, add_to!, sub_to!, mul_to!, div_to!

const IterType = Union{NTuple{N,Tuple} where N, Vector, Matrix, Base.Iterators.Repeated}

# Functions from ROIViews:

"""
    expand_add(t1,t2)  # adds t1 to t2 as a tuple and returns t2[n] for n > length(t1)

adds the elements of the tuple t1 to t2 for components in t1 which exist for the rest just t2 is used.

Example:
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

expands a size tuple sz with the sizes as given in the tuple sz2 for positions which do not exist in sz. Typically one wants to
obtain a tuple, which is achieved by a wrapping cast: `Tuple(expand_size(sz,sz2))`

Example:
```jldoctest
julia> sz = expand_size((1,2,3),(4,5,6,7,8,9))
Base.Generator{UnitRange{Int64}, var"#7#8"{Tuple{Int64, Int64, Int64}, NTuple{6, Int64}, Int64}}(var"#7#8"{Tuple{Int64, Int64, Int64}, NTuple{6, Int64}, Int64}((1, 2, 3), (4, 5, 6, 7, 8, 9), 3), 1:6)

julia> Tuple(sz)
(1, 2, 3, 7, 8, 9)
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

Example:
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

Example:
```jldoctest
julia> g = curry(+,10.0)
#9 (generic function with 1 method)

julia> g(3)
13.0
```
"""
curry(f, x) = (xs...) -> f(x, xs...)   # just a shorthand to remove x

## These functions ensure that also numbers can be iterated and zipped
"""
    cast_iter(vals::Matrix)

a conveniance function used to make number itarable via repeated. This is used when dealing with multiple arguments where one can be optionally an iterable.
A Tuple is also interpreted as repeating this tuple over and over.
"""
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

"""
    cast_number_iter(vals::Matrix)

a conveniance function used to make number itarable via repeated. This is used when dealing with multiple arguments where one can be optionally an iterable.
The behavious is slightly different to cast_iter when it comes to Tuples
A Tuple is here iterated as a list of numbers.
"""
function cast_number_iter(vals::Vector{<:Number})
    Tuple(vals)   # makes the matrix iterable, interpreting it as a series of vectors
end

function cast_number_iter(vals::NTuple{N,<:Number} where N)
    vals
end

function cast_number_iter(vals::Number)
    repeated(vals) # during the zip operation this type always yields the same value
end

"""
    optional_mat_to_iter(vals)  # only for matrices

casts a matrix via cast_iter, all other types are unaffected
"""
function optional_mat_to_iter(vals)  # only for matrices
    vals
end

function optional_mat_to_iter(vals::Matrix)
    cast_iter(vals)
end

"""
    mat_to_tvec(v) 
converts a 2d matrix to a Vector of Tuples

Example:
```jldoctest
julia> NDTools.mat_to_tvec([1 3 5; 2 4 6])
3-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (3, 4)
 (5, 6)
```
"""
mat_to_tvec(v) = [Tuple(v[:,n]) for n in 1:size(v,2)] # converts a 2d matrix to a Vector of Tuples

"""
    apply_dims(scale, dims, N)

replaces scale entries not in dims with zeros. Can also be applied to an Iterable of Tuples

Arguments:
+ scale: Vector to pass its valid entries through 
+ dims: Tuple of valid dimensions
+ N: total length of the resulting  `NTuple`

Example:
```jldoctest
julia> NDTools.apply_dims((1,2,3),(1,2), 4)
(1, 2, 0, 0)
```
"""
function apply_dims(scale, dims, N)  # replaces scale entries not in dims with zeros
    ntuple(i -> i ∈ dims ? scale[i] : zero(eltype(scale)), N)
end

function apply_dims(scales::IterType, dims, N)  # replaces scale entries not in dims with zeros
    Tuple(ntuple(i -> i ∈ dims ? scale[i] : zero(eltype(scale)), N)  for scale in  scales)
end

# Functions for IndexFunArrays.utils

"""
    single_dim_size(dim::Int,dim_size::Int, tdim=dim)

Returns a tuple (length `tdim`, which by default is dim) of singleton sizes except at the final position `dim`, which contains `dim_size`

Arguments:
+ dim: non-zero position
+ dim_size: the value this non-zero position is given in the returned NTuple
+ tdim: total length of the returned NTuple

Example
```jldoctest
julia> single_dim_size(4, 3)
(1, 1, 1, 3)

julia> single_dim_size(4, 5)
(1, 1, 1, 5)

julia> single_dim_size(2, 5)
(1, 5)
```
"""
function single_dim_size(dim::Int,dim_size::Int,tdim=dim)
    Base.setindex(Tuple(ones(Int, tdim)),dim_size,dim)
end

"""
    reorient(vec, dim)

reorients a 1D vector `vec` along dimension `dim`.
"""
function reorient(vec, dim::Int)
    reshape(vec, single_dim_size(dim, length(vec)))
end

"""
    collect_dim(col, dim::Int)

collects a collection `col` and reorients it into direction `dim`.

Example:
 ```jldoctest
julia> collect_dim(1:5,2)
1×5 Matrix{Int64}:
 1  2  3  4  5
```
"""
function collect_dim(col, dim::Int)
    reorient(collect(col), dim)
end

"""
    linear_index(pos, sz)

converts a tuble (pos) to a linear index using the size (sz).

Example:
```doctest
julia> a = rand(10,10,10);

julia> linear_index((3,4,5), (10,10,10))
433

julia> a[3,4,5]  == a[433]
true
```
"""
function linear_index(pos, sz)
    factors = (1, cumprod(sz[1:end-1])...)
    sum((pos.-1) .* factors)+1
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

"""
    apply_tuple_list(f, t1,t2)

applies a two-argument function to tubles and iterables of tuples, if either of the arguments `t1` or `t2` is an Iterable
"""
function apply_tuple_list(f, t1, t2)  # applies a two-argument function to tubles and iterables of tuples
    return f(t1,t2)
end

function apply_tuple_list(f, t1, t2::IterType)
    return Tuple([f(t1,a2) for a2 in t2])
end

function apply_tuple_list(f, t1::IterType, t2)
    res= Tuple([f(a1,t2) for a1 in t1])
    return res
end

function apply_tuple_list(f, t1::IterType, t2::IterType)
    return Tuple([f(a[1],a[2]) for a in zip(t1,t2)])
end


## Functions from FourierTools utils:

"""
    select_sizes(x::AbstractArray, dim; keep_dims=true)

Additional size method to access the size at several dimensions
in one call.
`keep_dims` allows to return the other dimensions as singletons.

Examples
```jldoctest
julia> x = ones((2,4,6,8, 10));

julia> select_sizes(x, (2,3))
(1, 4, 6, 1, 1)

julia> select_sizes(x, 5)
(1, 1, 1, 1, 10)

julia> select_sizes(x, (5,))
(1, 1, 1, 1, 10)

julia> select_sizes(x, (2,3,4), keep_dims=false)
(4, 6, 8)

julia> select_sizes(x, (4,3,2), keep_dims=false)
(8, 6, 4)
```

"""
function select_sizes(x::AbstractArray{T},dim::NTuple{N,Int};
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

function select_sizes(x::AbstractArray, dim::Integer; keep_dims=true)
    select_sizes(x, Tuple(dim), keep_dims=keep_dims)
end


"""
    slice(arr, dim, index)

Return a `N` dimensional slice (where one dimensions has size 1) of the N-dimensional `arr` at the index position
`index` in the `dim` dimension of the array.
It holds `size(out)[dim] == 1`.

Examples
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

Arguments:
`a` should be the axes obtained by `axes(arr)` of an array.
`dim` is the dimension to be selected and `index` the index of it.

Examples
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
    expand_dims(x, ::Val{N})
    expand_dims(x, N::Number)

expands the dimensions of an array to a given number of dimensions.

Try to prefer the `Val` version because this is type-stable.
`Val(N)` encapsulates the number in a type from which the compiler
can then infer the return type.

Examples
The result is a 5D array with singleton dimensions at the end
```jldoctest
julia> expand_dims(ones((1,2,3)), Val(5))
1×2×3×1×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 1.0  1.0

[:, :, 2, 1, 1] =
 1.0  1.0

[:, :, 3, 1, 1] =
 1.0  1.0

julia> expand_dims(ones((1,2,3)), 5)
1×2×3×1×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 1.0  1.0

[:, :, 2, 1, 1] =
 1.0  1.0

[:, :, 3, 1, 1] =
 1.0  1.0
```
"""
function expand_dims(x, N::Number)
    return reshape(x, (size(x)..., ntuple(x -> 1, (N - ndims(x)))...))
end

function expand_dims(x, ::Val{N}) where N
    return reshape(x, (size(x)..., ntuple(x -> 1, (N - ndims(x)))...))
end

"""
    flatten_trailing_dims(arr, max_dim=length(arr)÷2+1)

flattens (squeezes) the trailing dims. `max_dim` denotes the last dimension to keep. The implementation
uses reshape and thus returns a modified view of the array referring to the same data. 
By default max_dim is adjusted such that a 2N array is squeezed into an N+1 array as needed for a scan.
"""
function flatten_trailing_dims(arr, max_dim)
    reshape(arr,(size(arr)[1:max_dim-1]...,prod(size(arr)[max_dim:end])))
end

"""
    regular_pattern(sz, offset=0, step=1)

returns a generator with tuples that point to a regular grid pattern.
Note that the result is zero-based, which means you will need to add `1` to each tuple element to use this for indexing into arrays.

Arguments:
+ sz: size of the underlaying array for which to generate the regular pattern
+ offset: offset of the first position. Can be tuple or scalar.
+ step: step between the indices. Can be tuple or scalar.
"""
function regular_pattern(sz, offset=0, step=1) # zero-based
    return (offset.+step.*(Tuple(pos).-1) for pos in CartesianIndices(1 .+(((sz.-1).-offset) .÷ step)))
end

"""
    get_scan_pattern(sz, pitch=1, step=1; dtype=Float32, flatten_scan_dims=false)

generates a scan pattern in N dimensions based on scanning an array of size `sz`, with a scan pitch of `pitch` and stepping by `step`.
The result is an array with 2*length(sz) dimensions.
Note that the scan needs to be commensurate, implying that `pitch` is an integer multiple of `step`.
"""
function get_scan_pattern(sz, pitch=1, step=1; dtype=Float32, flatten_scan_dims=false)
    if any(pitch .% step .!= 0)
        error("Scan pitch needs to be commensurate with scan step")
    end
    @show scan_sz = pitch .÷ step;
    res = zeros(dtype, (sz...,scan_sz...))
    for scan_pos in regular_pattern(scan_sz,0,1)
        for pos in regular_pattern(sz,step.*scan_pos,pitch)
            res[(1 .+ pos)...,(1 .+ scan_pos)...] = one(dtype)
        end
    end
    if flatten_scan_dims
        return flatten_trailing_dims(res, length(sz)+1);
    else
        return res
    end
end

"""
    image_to_arr(img)

converts an images as obtained by the `testimage` function into an array.
"""
function image_to_arr(img)
    return Float32.(permutedims(img,(2,1)))
end

"""
    center_position(field)

position of the center of the `field` according to the Fourier-space nomenclature (pixel right of center for even-sized arrays).
returns a tuple of ints which can be used for indexing.
"""
function center_position(field)
    (size(field) .÷ 2).+1
end

"""
    center_value(field)

value of the `field` at the position of the center of the `field` according to the Fourier-space nomenclature (pixel right of center for even-sized arrays).
"""
function center_value(field)
    field[center_position(field)...]
end

"""
    ft_center_diff(s [, dims])

Calculates how much each dimension must be shifted that the
center frequency is at the Fourier center.
This if for a normal `fft`
"""
function ft_center_diff(s::NTuple{N, T}, dims=ntuple(identity, Val(N))) where {N, T}
    ntuple(i -> i ∈ dims ?  s[i] ÷ 2 : 0 , N)
end


"""
    select_region(mat;new_size=size(mat), center=ft_center_diff(size(mat)).+1, pad_value=zero(eltype(mat)))

selects (extracts) a region of interest (ROI), defined by `new_size` and centered at `center` in the source image. Note that
the number of dimensions can be smaller in `new_size` and `center`, in which case the default values will be insterted
into the missing dimensions. `new_size` does not need to fit into the source array and missing values will be replaced with `pad_value`.

Arguments:
+ `new_size`. The size of the array view after the operation finished. By default the original size is assumed
+ `center`. Specifies the center of the new view in coordinates of the old view. By default an alignment of the Fourier-centers is assumed.
+ `pad_value`. Specifies the value which is inserted in case the ROI extends to outside the source area.

The returned results is a mutable view, which allows this method to also be used for writing into a ROI

Examples
```jldoctest
julia> using NDTools

julia> select_region(ones(3,3),new_size=(7,7),center=(1,3))
7×7 PaddedView(0.0, OffsetArray(::Matrix{Float64}, 4:6, 2:4), (Base.OneTo(7), Base.OneTo(7))) with eltype Float64:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
function select_region(mat; new_size=size(mat), center=ft_center_diff(size(mat)).+1, pad_value=zero(eltype(mat)))
    new_size = Tuple(expand_size(new_size, size(mat)))
    center = Tuple(expand_size(center, ft_center_diff(size(mat)).+1))
    oldcenter = ft_center_diff(new_size).+1
    MutablePaddedView(PaddedView(pad_value, mat,new_size, oldcenter .- center.+1));
end

"""
    get_src_dst_range(src_size, dst_size, new_size, center)

A helpfer function to calculate the index ranges to copy from source size `src_size` to destination size `dst_size` with the
integer center position of the destination aligning with the position in the source as specified by center.
"""
function get_src_dst_range(src_size, dst_size, new_size, src_center, dst_ctr=dst_size .÷2 .+1)
    ROI_center = (new_size.÷2 .+1)
    src_start = src_center .- ROI_center  .+1 # start of the first pixel to copy (without clipping)
    src_end = src_start .+ new_size .- 1 # the last pixel to copy
    src_start_clip = max.(1, src_start)
    src_end_clip = min.(src_end, src_size)
    if any(src_start_clip .> src_size) || any(src_end_clip .< 1)
        return (1:0), (1:0)
    end
    extra_src_start = max.(0, src_start_clip .- src_start)
    extra_src_end = max.(0, src_end .- src_end_clip)
    copy_size = new_size .- extra_src_start .- extra_src_end

    dst_start = dst_ctr .- ROI_center .+1 .+ extra_src_start
    dst_end = dst_start .+ copy_size .- 1
    dst_end_clip = min.(dst_end, dst_size)
    dst_start_clip = max.(1, dst_start)
    if any(dst_start_clip .> dst_size) || any(dst_end_clip .< 1)
        return (1:0), (1:0)  # returns an empty range for all coordinates
    end

    extra_dst_start = max.(0, dst_start_clip .- dst_start)
    extra_dst_end = max.(0, dst_end .- dst_end_clip)
    src_start_clip = src_start_clip .+ extra_dst_start
    src_end_clip = max.(0, src_end_clip .- extra_dst_end)

    range_src = Tuple((src_start_clip[d]:src_end_clip[d]) for d in 1:length(src_start))
    range_dst = Tuple((dst_start_clip[d]:dst_end_clip[d]) for d in 1:length(dst_start))
    return range_src, range_dst
end

"""
    assign_to!(a,b)
    
assignes array (or value) `b` to array `a` pointwise.
Helper function to be passed as an operator to functions such as select_region!
"""
function assign_to!(a,b)
    a .= b
end

"""
    add_to!(a,b)

adds array (or value) `b` to array `a` pointwise.
Hlper function to be passed as an operator to functions such as select_region!
"""
function add_to!(a,b)
    a  .+= b
end

"""
    sub_to!(a,b)
    
subtractss array (or value) `b` from array `a` pointwise and assigns into a.
Hlper function to be passed as an operator to functions such as select_region!
"""
function sub_to!(a,b)
    a  .-= b
end

"""
    mul_to!(a,b)

multiplies array (or value) `b` to array `a` pointwise.
Helper function to be passed as an operator to functions such as select_region!
"""
function mul_to!(a,b)
    a .*= b
end

"""
    div_to!(a,b)
    
divides array `a` by array (or value) `b` to array a pointwise.
Helper function to be passed as an operator to functions such as select_region!
"""
function div_to!(a,b)
    a ./= b
end


"""
    select_region!(src, dst=nothing, new_size=size(src), center=size(src).÷2 .+1, dst_center=nothing, pad_value=zero(eltype(mat), operator!=assign_to!))

selects (extracts, pads, shifts) a region of interest (ROI), defined by `new_size` and centered with the destination center aligned at 
the position `center` in the source image. Note that the number of dimensions in `new_size`,  `center` and `dst_center` can be smaller , 
in which case default values (see below) will be insterted into the missing dimensions. `new_size` does not need to fit into the source array 
and missing values will be replaced with `pad_value`, if no `dst` is provided.

As opposed to `select_region()`, this version returns a copy rather than a view or, alternatively, also writes into a destination array `dst` 
(`new_size` is then interpreted to refer to the maximally assigned region). 
If `nothing` is provided for `dst`, a new array of size `new_size` is created.

Arguments:
+ `src`. The source array to select from.
+ `dst`. The destination array to write into, if provided. By default `dst=nothing` a new array is created. The `dst`array (or new array) is returned. 
+ `new_size`. The size of the array view after the operation finished. By default the original size is assumed
+ `center`. Specifies the center of the new view in coordinates of the old view. By default an alignment of the Fourier-center (right center) is assumed.
+ `dst_center`. defines the center coordinate in the destination array which should align with the above source center. If nothing is provided, the right center pixel of the `dst` array or new array is used.
+ `pad_value`. specifies the value which is inserted in case the ROI extends to outside the source area. This is only used, if no `dst` array is provided.
+ `operator!`. allows to provide a user_defined array assignment function. The function my_op!(dst,src) should operator on array views and typically perform the assignment elementwise, overwriting the entries in dst.
                Five such functions are exported by NDTools: `assign_to!`, `add_to!`, `sub_to!`, `mul_to!`, `div_to!`, representing the operations `.=`, `.+=`, `.-=`, `.*=` and `./=` respectively.

The returned results is the destination (or newly created) array.
Note that this version is rather fast, since it consists of only a sinlge sub-array assigment on views, avoiding copy operations.

Examples:
```jdoctest
julia> a = ones(5,6);

julia> select_region!(a,new_size=(10,10))  # pad a with zeros to a size of (10,10)
10×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> dst=elect_region!(a,new_size=(10,10), dst_center=(1,1)) # pad a with zeros to a size of (10,10), but place original center at the corner
10×10 Matrix{Float64}:
 1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

 julia> select_region!(2 .*a,dst, dst_center=size(dst)) # write a doubled version into the bottom right corner
10×10 Matrix{Float64}:
 1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  2.0  2.0  2.0  2.0
 0.0  0.0  0.0  0.0  0.0  0.0  2.0  2.0  2.0  2.0
 0.0  0.0  0.0  0.0  0.0  0.0  2.0  2.0  2.0  2.0
```
"""
function select_region!(src::T, dst=nothing; new_size=nothing, 
                        center=size(src).÷2 .+1, dst_center=nothing, pad_value=zero(eltype(src)), 
                        operator! =assign_to!) where {T}
    pad_value = eltype(T)(pad_value)

    new_size = let 
        if isnothing(new_size)
            if isnothing(dst)
                size(src)
            else
                size(dst)
            end
        else
            new_size
        end
    end

    new_size = let 
        if isnothing(dst)
            new_size = Tuple(expand_size(new_size, size(src)))
        else
            new_size = Tuple(expand_size(new_size, size(dst)))
        end
    end

    
    dst = let 
        if isnothing(dst)
            if isnothing(new_size)
                dst=fill(pad_value,size(src))
            else
                dst=fill(pad_value,new_size)
            end
        else
            dst
        end
    end

    dst_center = isnothing(dst_center) ? size(dst).÷ 2 .+1 : dst_center
    
    center = Tuple(expand_size(center, size(src).÷2 .+1))
    dst_center = Tuple(expand_size(dst_center, size(dst).÷ 2 .+1))

    range_src, range_dst = get_src_dst_range(size(src),size(dst),new_size,center, dst_center)
    if !isempty(range_dst)
        v_src = @view src[range_src...]
        v_dst = @view dst[range_dst...]
        operator!(v_dst, v_src)  # for some strange reason this is faster (and of course more flexible) than the line below.
        # dst[range_dst...] .+= src[range_src...]
    end
    return dst::T
end

struct MutablePaddedView{T,N,I,A} <: AbstractArray{T,N}
    data::PaddedView{T,N,I,A}
end

Base.size(A::MutablePaddedView) = size(A.data)
# Base.ndims(A::MutablePaddedView) = ndims(A.data)

# The change below makes select_region into a writable region
Base.@propagate_inbounds function Base.setindex!(A::MutablePaddedView{T,N}, v, pos::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A.data, pos...)
    if Base.checkbounds(Bool, A.data.data, pos...)
        return setindex!(A.data.data, v, pos... )
    else
        return A.data.fillvalue
    end
end

Base.@propagate_inbounds function Base.getindex(A::MutablePaddedView{T,N}, pos::Vararg{Int,N}) where {T,N}
    getindex(A.data, pos...)
end


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

"""
    idx_to_dim(idx_arr,dim=ndims(idx_arr)+1)  # this should be a view

converts an N-dimensional array of NTuple to an N+1 dimensional array.
This should eventually be realsized as a view rather than a copy operation.

Arguments:
+ `idx_arr`. The array of NTuple to convert
+ `dims`. Optional argument for the destination direction. The default is to append (stack) one dimension.

"""
function idx_to_dim(idx_arr, dim=ndims(idx_arr)+1)  # this should be a view
    cat((getindex.(idx_arr,d) for d in 1:length(idx_arr[1]))..., dims=dim)
end

"""
    ϕ_tuple(t::NTuple)

helper function to obtain the angle from a tuple of coordinates. The azimuthal angle is calculated via the `atan`. However, the order of the NTuple is (x,y).

Arguments:
+ `t`: an NTuple. Only the first two coordinates (x,y) are used and the azimuthal angle ϕ is returned
"""
ϕ_tuple(t::NTuple) = atan(t[2],t[1])



"""
    pack(myTuple::Tuple, do_fit)

this packs a tuple of values into a vector which is normalized per direction and returns an unpack function which reverts this.
This tool is useful for fit routines.

returns the packed tuples (with a position being true in do_fit) as a vector and the unpack algorithm as a closure.
"""
function pack(myTuple::Tuple, do_fit; rel_scale=nothing, dtype=Float64)
    scales = []
    vec = dtype[]
    lengths = Int[]
    for (t,f) in zip(myTuple,do_fit)
        if !isnothing(rel_scale)
            scale = sum(t)/length(t) .* rel_scale
        else
            scale = 1.0
        end
        if f
            push!(scales, scale)
            vec = cat(vec, t ./ scale, dims=1)
            push!(lengths, length(t))
        else # save the constant
            push!(scales, t)
            push!(lengths, length(t))
        end
    end

    function unpack(vec)
        res = ()
        pos = 1
        for (l,s,f) in zip(lengths,scales, do_fit)
            if f
                vals = vec[pos:pos+l-1] .* s
                pos += l
            else
                vals = s
            end
            res = (res..., vals)
        end
        return res
    end
    
    return Vector{Float64}(vec), unpack # Vector{Int}(lengths), Vector{Float64}(scales) 
end

"""
    soft_theta(val, eps=0.1) = (val .> eps) ? 1.0 : ((val .< -eps) ? 0.0 : (1.0 .- cos((val .+ eps).*(pi/(2*eps))))./2) # to make the threshold differentiable

this is a version of the theta function that uses a soft transition and is differentialble.

Arguments:
+ val: value to compare with zero
+ eps: hardness of the step function (spanning from -eps to eps)
"""
soft_theta(val, eps=0.01) = (val .> eps) ? 1.0 : ((val .< -eps) ? 0.0 : (1.0 .- cos.((val .+ eps).*(pi/(2*eps))))./2.0) # to make the threshold differentiable

"""
    soft_delta(val, eps=0.1) = (val .> eps) ? 1.0 : ((val .< -eps) ? 0.0 : (1.0 .- cos((val .+ eps).*(pi/(2*eps))))./2) # to make the threshold differentiable

this is a smooth version of the theta function that uses a soft peak and is differentialble.
The sum is not normalized but the value at val is one.

Arguments:
val: value to compare with zero
eps: hardness of the step function (spanning from -eps to eps)
"""
soft_delta(val, eps=0.01) = (abs2.(val) .> abs2.(eps)) ? 0.0 : (1.0 .+ cos.(val.*(pi/eps)))./2.0 # to make the threshold differentiable

"""
    exp_decay(t,τ, eps=0.1) 

an exponential decay starting at zero and using a soft threshold.
Note that this function can be applied to multiple decay times τ simulataneously, yielding multiple decays stacked along the second dimension
"""
exp_decay(t,τ, eps=0.01) = soft_theta.(t,eps) .* exp.( - (t ./ transpose(τ)))


"""
    multi_exp_decay(t,amps, τs, eps=0.1) 

a sum of exponential decays starting at t==zero and using a soft threshold.

Arguments:
+ t: time series to apply this to
+ amps: individual amplitudes as a vector
+ τs : individual lifetimes as a vector
+ eps: width of the soft edge
"""
multi_exp_decay(t, amps, τs, eps=0.01) = sum(transpose(amps).*exp_decay(t, τs, eps), dims=2)[:,1] 

"""
    radial_mean(data; maxbin=nothing, bin_step=nothing, pixelsize=nothing)

calculates the radial mean of a dataset `data`.
returns a tuple of the radial_mean and the bin_centers.

Arguments:
+ data: data to radially average
+ maxbin: a maximum bin value
+ bin_step: optionally defines the step between the bins
"""
function radial_mean(data; nbins=nothing, bin_step=nothing, offset=CtrFT, scale=nothing)
    if isnothing(scale)
        scale=Tuple(ones(Int,ndims(data)))
    end
    if isnothing(bin_step)
        bin_step=maximum(scale)
    end
    
    idx = round.(Int, rr(size(data), offset=offset, scale=scale./bin_step)) .+ 1
    if isnothing(nbins)
        nbins = maximum(idx)
    else
        idx[idx.>nbins] .= nbins
    end

    bincenters = bin_step .* collect(0.5:nbins-0.5)

    myevents = zeros(Float64, nbins)
    mysum = zeros(Float64, nbins)
    idxlist = idx[:]
    v = @view mysum[idxlist]
    v .+= data[:]
    v = @view myevents[idxlist]
    v .+= 1
    myevents[myevents .== 0] .= 1
    return mysum ./ myevents, bincenters
end

"""
    Δ_phase(arr, dim)

calculates the relative phase slope along dimension `dim` of a non-zero array `arr` without wrap-around problems.

Arguments:
+ arr: array of which to evaluate the phase slope
+ dim: dimension along which to evaluate the slope
"""
function Δ_phase(arr, dim)
    no_shift = ((n==dim) ? (1:size(arr,dim)-1) : (:) for n in 1:ndims(arr))
    shift = ((n==dim) ? (2:size(arr,dim)) : (:) for n in 1:ndims(arr))
    return angle.(arr[no_shift...] ./ arr[shift...]) # this is a complex-valued division to obtain the relative phase angle!
end


function moment_proj(data, h=3; pdims=3)
    mdata = mean(data, dims=pdims)
    return mean((data .- mdata).^h, dims=pdims)
end

"""
    moment_proj_normed(data, h=3; pdims=3)

performes a projection over the `h`-th moment of the data along dimension(s) `pdims` normed by the variance.
"""
function moment_proj_normed(data, h=3; pdims=3)
    moment_proj(data, h, pdims=pdims) ./ (moment_proj(data,2, pdims=pdims) .^((h-1)/2))
end


end # module
