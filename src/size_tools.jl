# size_tools.jl

export collect_dim, select_sizes, select_sizes_squeeze, center_position, center_value
export expand_add, expand_size, optional_posZ, reorient, single_dim_size

"""
    contains all functions that operate on Tuples, sizes and alike
"""

# Functions from ROIViews:

"""
    expand_add(t1,t2)  # adds t1 to t2 as a tuple and returns t2[n] for n > length(t1)

Adds the elements of the tuple `t1` to `t2`.
If `t1` is shorter than `t2`, we take the tail of `t2` as elements.
Output length is always the same length as `t2`

Example:
```jldoctest
julia> expand_add((1,2,3),(4,5,6,7,8,9))
(5, 7, 9, 7, 8, 9)

julia> expand_add((1,1), (0,))
(1,)

julia> expand_add((1,2), (0,0))
(1, 2)

julia> expand_add((1,2), (0,0,0))
(1, 2, 0)
```
"""
function expand_add(t1::NTuple{M, T1},t2::NTuple{N, T2}) where {M, N, T1, T2} 
    return ntuple(i -> i ≤ M ? t1[i] + t2[i] : t2[i], Val(N))
end

"""
    expand_size(sz,sz2)

Expands a size tuple `sz` with the sizes as given in the tuple `sz2` for positions which do not exist in `sz`. 
Typically one wants to

Example:
```jldoctest
julia> sz = expand_size((1,2,3),(4,5,6,7,8,9))
Base.Generator{UnitRange{Int64}, var"#7#8"{Tuple{Int64, Int64, Int64}, NTuple{6, Int64}, Int64}}(var"#7#8"{Tuple{Int64, Int64, Int64}, NTuple{6, Int64}, Int64}((1, 2, 3), (4, 5, 6, 7, 8, 9), 3), 1:6)

julia> Tuple(sz)
(1, 2, 3, 7, 8, 9)
```
"""
function expand_size(sz::NTuple{M, T1},sz2::NTuple{N, T2}) where {M, N, T1, T2}
    return ntuple(i -> i ≤ M ? sz[i] : sz2[i], Val(N))
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

# Functions for IndexFunArrays.utils

"""
    single_dim_size(dim::Int, dim_size::Int, tdim=dim)

Returns a tuple (length `tdim`, which by default is `dim`) of singleton sizes 
except at the final position `dim`, which contains `dim_size`.

!!! warning "Not type-stable!"
    Is not type stable!


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

julia> single_dim_size(3,5, 4)
(1, 1, 5, 1)
```
"""
function single_dim_size(dim::Int, dim_size::Int, tdim=dim)
    Base.setindex(ntuple(i -> 1, Val(tdim)), dim_size, dim)::NTuple{tdim, Int}
end

"""
    single_dim_size(::Val{dim}, dim_size::Int, tdim=::Val{dim})

Same `single_dim_size` but type stable with `Val`.
"""
function single_dim_size(::Val{dim}, dim_size::Int, tdim=Val(dim)) where dim
    Base.setindex(ntuple(i -> 1, tdim), dim_size, dim)
end


"""
    reorient(vec, dim)

Reorients a 1D vector `vec` along dimension `dim`.

!!! warning "Not type-stable!"
    Is not type stable!

```julia
julia> reorient([1,2,3,4], 3)
1×1×4 Array{Int64, 3}:
[:, :, 1] =
 1

[:, :, 2] =
 2

[:, :, 3] =
 3

[:, :, 4] =
 4
```
"""
function reorient(vec, dim::Int)
    reshape(vec, single_dim_size(dim, length(vec)))
end

"""
    reorient(vec, ::Val{dim})

Type stable version of `reorient`!

```jldoctest
reorient([1,2,3,4], Val(3))
1×1×4 Array{Int64, 3}:
[:, :, 1] =
 1

[:, :, 2] =
 2

[:, :, 3] =
 3

[:, :, 4] =
 4
```
"""
function reorient(vec, ::Val{dim}) where dim
    reshape(vec, single_dim_size(Val(dim), length(vec)))
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
    collect_dim(col, dim::Val)

Type stable version!

```jldoctest
julia> collect_dim(1:5, Val(2))
1×5 Matrix{Int64}:
 1  2  3  4  5
```

"""
function collect_dim(col, dim::Val)
    reorient(collect(col), dim)
end



## Functions from FourierTools utils:

"""
    select_sizes(x::AbstractArray, dim)

Additional size method to access the size at several dimensions
in one call.
Keep singleton dimensions.

Examples
```jldoctest
julia> x = ones((2,4,6,8, 10));

julia> select_sizes(x, (2,3))
(1, 4, 6, 1, 1)

julia> select_sizes(x, 5)
(1, 1, 1, 1, 10)

julia> select_sizes(x, (5,))
(1, 1, 1, 1, 10)
```
"""
function select_sizes(x::AbstractArray{T, M}, dim::NTuple{N,Int}) where {T,N,M}
    sz = ntuple(i -> 1, Val(M))
    for n in dim
        sz = Base.setindex(sz, size(x, n), n)
    end
    return Tuple(sz)
end 


function select_sizes(x::AbstractArray, dim::Integer)
    select_sizes(x, (dim,))
end

"""
    select_sizes_squeeze(x::AbstractArray, dim; keep_dims=true)

Additional size method to access the size at several dimensions
in one call.
Remove singleton dimensions.

See also [`select_sizes`](@ref select_sizes) which does not remove singleton dimensions.

Examples
```jldoctest
julia> select_sizes_squeeze(randn((5,6,7)), (2,3))
(6, 7)

julia> select_sizes_squeeze(randn((5,6,7)), 2)
(6,)
```
"""
function select_sizes_squeeze(x::AbstractArray{T, M}, dim::NTuple{N,Int}) where {T,N,M}
    return map(n -> size(x,n), dim)
end 


function select_sizes_squeeze(x::AbstractArray, dim::Integer)
    select_sizes_squeeze(x, (dim,))
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


function bin(arr, factors=Tuple(2 .*ones(Int64,ndims(arr))))
    sz = size(arr) # old size
    sn = sz .÷ factors # new size
    s2 = Tuple((isodd(i) ? factors[(i+1)/2] : sn[i/2] for i in 1:2*length(sz)))
    dims = Tuple(2 .*(1:length(sz)) .- 1)
    reshape(sum(reshape(arr,s2),dims=dims),sn)
end
