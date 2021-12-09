## iteration_tools.jl
## These functions ensure that also numbers can be iterated and zipped
export linear_index
export apply_tuple_list

"""
    cast_iter(vals::Matrix)

A convenience function used to make number iterable via repeated. 
This is used when dealing with multiple arguments where one can be 
optionally an iterable.
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

