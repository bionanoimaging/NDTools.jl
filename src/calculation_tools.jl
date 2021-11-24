## calculatoin_tools.jl
export pack
export radial_mean, Δ_phase
export moment_proj_normed

"""
    radial_mean(data; maxbin=nothing, bin_step=nothing, pixelsize=nothing)

Calculates the radial mean of a dataset `data`.
Returns a tuple of the radial_mean and the bin_centers.

 ## Arguments:
+ `data`: data to radially average
+ `maxbin`: a maximum bin value
+ `bin_step`: optionally defines the step between the bins


## Examples
```julia-repl
julia> radial_mean([5, 1, 1, 3, 15])
([1.0, 2.0, 10.0], [0.5, 1.5, 2.5])
```
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

Calculates the relative phase slope along dimension `dim` of a non-zero array `arr` without wrap-around problems.

Arguments:
+ arr: array of which to evaluate the phase slope
+ dim: dimension along which to evaluate the slope
"""
function Δ_phase(arr, dim)
    no_shift = ((n==dim) ? (1:size(arr,dim)-1) : (:) for n in 1:ndims(arr))
    shift = ((n==dim) ? (2:size(arr,dim)) : (:) for n in 1:ndims(arr))
    return angle.(arr[no_shift...] ./ arr[shift...]) # this is a complex-valued division to obtain the relative phase angle!
end

"""
    moment_proj(data, h=3; pdims=3)

Performs a projection over the `h`-th moment of the data along dimension(s) `pdims`.
See also [`moment_proj_normed`](@ref NDTools.moment_proj_normed)

## Example
```jldoctest
julia> NDTools.moment_proj([1 3; 2 4], 2, pdims=(1,))
1×2 Matrix{Float64}:
 0.25  0.25
```
"""
function moment_proj(data, h=3; pdims=3)
    mdata = mean(data, dims=pdims)
    return mean((data .- mdata).^h, dims=pdims)
end

"""
    moment_proj_normed(data, h=3; pdims=3)

Performs a projection over the `h`-th moment of the data along dimension(s) `pdims` normed by the variance.


## Example
```jldoctest
julia> NDTools.moment_proj_normed([1 3; 2 4], 2, pdims=(1,))
1×2 Matrix{Float64}:
 0.5  0
```
"""
function moment_proj_normed(data, h=3; pdims=3)
    moment_proj(data, h, pdims=pdims) ./ (moment_proj(data,2, pdims=pdims) .^((h-1)/2))
end
