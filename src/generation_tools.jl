## generation_tools.jl
export get_scan_pattern, ϕ_tuple, idx_to_dim


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
    idx_to_dim(idx_arr)  # this should be a view

Converts an N-dimensional array of NTuple to an N+1 dimensional array by orienting the (inner) tuple along the (outer) trailing+1 dimension.

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
function idx_to_dim(idx_arr::AbstractArray{T, N}) where {T, N} 
    out_arr = Array{typeof(idx_arr[1][1])}(undef, size(idx_arr)..., length(idx_arr[begin]))

    # loop over array
    @inbounds for I in CartesianIndices(idx_arr)
        for (i, ind) in enumerate(axes(out_arr, N+1))
            out_arr[Tuple(I)..., ind] = idx_arr[I][i]
        end
    end
    return out_arr
end
