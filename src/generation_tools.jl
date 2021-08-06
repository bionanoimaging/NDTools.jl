## generation_tools.jl
export get_scan_pattern, ϕ_tuple, soft_theta, exp_decay, multi_exp_decay, soft_delta, idx_to_dim, image_to_arr

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

## conversion_tools.jl

"""
    idx_to_dim(idx_arr,dim=ndims(idx_arr)+1)  # this should be a view

converts an N-dimensional array of NTuple to an N+1 dimensional array by orienting the (inner) tuple along the (outer) trailing dimension.
This should eventually be realsized as a view rather than a copy operation.

Arguments:
+ `idx_arr`. The array of NTuple to convert
+ `dims`. Optional argument for the destination direction. The default is to append (stack) one dimension.

"""
function idx_to_dim(idx_arr, dim=ndims(idx_arr)+1)  # this should be a view
    cat((getindex.(idx_arr,d) for d in 1:length(idx_arr[1]))..., dims=dim)
end


"""
    image_to_arr(img)

converts an Image as obtained by the `testimage` function into an array.
"""
function image_to_arr(img)
    return Float32.(permutedims(img,(2,1)))
end

