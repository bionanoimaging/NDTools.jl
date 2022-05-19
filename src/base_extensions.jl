export sumdropdims

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
