export sumdropdims, insertdims


"""
    sumdropdims(arr; dims)

Alias for `dropdims(sum(arr, dims=dims), dims=dims)`
"""
function sumdropdims(arr::AbstractArray; dims)
    dropdims(sum(arr, dims=dims), dims=dims)
end


"""
    sumdropdims(f, arr; dims)

Alias for `dropdims(sum(f, arr, dims=dims), dims=dims)`
"""
function sumdropdims(f, arr::AbstractArray; dims)
    dropdims(sum(f, arr, dims=dims), dims=dims)
end


function insertdims(arr::AbstractArray{T, N}, dims::Int) where {T, N}
    t = ntuple(N+1) do i
        if i < dims
            size(arr, i)
        elseif i > dims
            size(arr, i-1)
        else
            1
        end
    end
    return reshape(arr, t...)::AbstractArray{T, N+1}
end
