## generation_tools.jl
export ϕ_tuple, idx_to_dim 

"""
    ϕ_tuple(t::NTuple)

Helper function to obtain the angle from a tuple of coordinates. 
The azimuthal angle is calculated via the `atan`. 
However, the order of the NTuple is (x,y).


Arguments:
+ `t`: an NTuple. Only the first two coordinates (x,y) are used and the azimuthal angle ϕ is returned
"""
ϕ_tuple(t::NTuple) = atan(t[2],t[1])

## conversion_tools.jl

"""
    idx_to_dim(idx_arr) 

Converts an N-dimensional array of NTuple to an N+1 dimensional array by orienting the (inner) tuple along the (outer) trailing+1 dimension.
Note that this function is not type stable!

See also: `idx_to_view` which reinterprets the array to an N+1 dimensional array by unrolling the (inner) tuple.
Arguments:
+ `idx_arr`. The array of NTuple to convert

Example:
```jldoctest
julia> idx_to_dim([(x,y) for x in 1:3, y in 1:3])
3×3×2 Array{Int64, 3}:
[:, :, 1] =
 1  1  1
 2  2  2
 3  3  3

[:, :, 2] =
 1  2  3
 1  2  3
 1  2  3
```
"""
function idx_to_dim(idx_arr::AbstractArray{T, N}) where {TT, NT, T<:NTuple{NT, TT}, N} 
    newdims = ntuple((d)->mod(d, N+1)+1, N+1)
    return permutedims(idx_to_arr_view(idx_arr), newdims)
end

"""
    dim_to_idx(v, ::Val{D}) where D 
converts an Array to an Array of Tuples by packaging the outer dimension into the inner dimension of the new array.

See also: `idx_to_dim` which is tarr_to_arr

Example:
```jldoctest
julia> NDTools.arr_to_tarr([1 3 5; 2 4 6])
3-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (3, 4)
 (5, 6)
```
"""
function dim_to_idx(v, ::Val{D}) where D
    newdims = ntuple((d)->mod(d-2, ndims(v))+1, ndims(v))
    return arr_to_idx_view(permutedims(v, newdims), Val(D))
end

"""
    idx_to_arr_view(idx_arr) 

Reinterprets an N-dimensional array of NTuple (idx_arr) as an N+1 dimensional array by unrolling the (inner) tuple.
Note that this function is not type stable!

See also: `arr_to_idx_view`

Arguments:
+ `idx_arr`. The array of NTuple to reinterpret

Example:
```jldoctest
julia> idx_to_arr_view([(x,y) for x in 1:3, y in 1:3])
2×3×3 reinterpret(reshape, Int64, ::Matrix{Tuple{Int64, Int64}}) with eltype Int64:
[:, :, 1] =
 1  2  3
 1  1  1

[:, :, 2] =
 1  2  3
 2  2  2

[:, :, 3] =
 1  2  3
 3  3  3
```
"""
function idx_to_arr_view(idx_arr::AbstractArray{T, N}) where {TT, NT, T<:NTuple{NT, TT}, N} 
    reinterpret(reshape, TT, idx_arr)
end

"""
    arr_to_idx_view(arr, ::Val{D}) where D 

Reinterprets an N-dimensional array as a N-1 dimensional array rolling the (inner) into a tuple.

See also: `idx_to_arr_view`

Arguments:
+ `arr`. The array to reinterpret
+ `D`. The Val type specifying the Tuple Length of the resulting array corresponding to the outermost dimension size

Example:
```jldoctest
julia> arr_to_idx_view([x for x in 1:3, y in 1:2], Val(3))
2-element reinterpret(reshape, Tuple{Int64, Int64, Int64}, ::Matrix{Int64}) with eltype Tuple{Int64, Int64, Int64}:
 (1, 2, 3)
 (1, 2, 3)
```
"""
function arr_to_idx_view(arr::AbstractArray{T, N}, ::Val{D}) where {T,N, D}
    reinterpret(reshape, NTuple{D, T}, arr)
end
