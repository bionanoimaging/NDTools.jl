## calculatoin_tools.jl
export pack
export radial_mean, Δ_phase
export moment_proj_normed

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
