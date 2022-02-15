export reverse_view

"""
    reverse_view(A::AbstractArray{T, N}; 
                 dims=ntuple(identity, Val(N))) where {T, N}

Creates a view of an array `A` which reverses all dimensions in `dims`.
Per default, `dims` is set to reverse all dimensions!
It is equivalent to `reverse(A, dims=dims)` but instead uses a view.

## Examples
```julia-repl
julia> A = [1 2 3; 4 5 6]
2×3 Matrix{Int64}:
 1  2  3
 4  5  6

julia> reverse_view(A, dims=1)
2×3 view(::Matrix{Int64}, 2:-1:1, 1:3) with eltype Int64:
 4  5  6
 1  2  3

julia> reverse(A, dims=1)
2×3 Matrix{Int64}:
 4  5  6
 1  2  3

julia> reverse_view(A, dims=(1,2))
2×3 view(::Matrix{Int64}, 2:-1:1, 3:-1:1) with eltype Int64:
 6  5  4
 3  2  1

julia> reverse(A, dims=(1,2))
2×3 Matrix{Int64}:
 6  5  4
 3  2  1
```
"""
function reverse_view(A::AbstractArray{T, N}; 
                      dims=ntuple(identity, Val(N))) where {T, N}
    # loop over all dimensions and put reversed or normal range
    out_inds = ntuple(Val(N)) do i
        if i ∈ dims
            lastindex(A, i):-1:firstindex(A, i)
        else
            # :1: is important so that both branches are StepRange{Int64, Int64}
            firstindex(A, i):1:lastindex(A, i)
        end
    end
    return view(A, out_inds...)::AbstractArray{T, N}
end
