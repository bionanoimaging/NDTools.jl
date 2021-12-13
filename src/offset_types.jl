export CenterFirst, CenterLast, CenterMiddle, CenterFT
export center

"""
    Center

Abstract supertype for all Center types.

See [`CenterFirst`](@ref), [`CenterLast`](@ref),
[`CenterMiddle`](@ref), [`CenterFT`](@ref).
"""
abstract type Center end 

"""
    CenterFirst

Type to indicate that the center should be 
at the first entry of the array.
This corresponds to the indices `(1,1,1,...)`.
"""
struct CenterFirst <: Center end


"""
    CenterEnd

Type to indicate that the center should be 
at the last entry of the array.
This corresponds in an array with size `(5,4,3)` to 
the indices `(5,4,3)`.
"""
struct CenterLast <: Center end


"""
    CenterMiddle

Type to indicate that the center should be 
at the mathematical center of the array (if interpreted
as a N dimensional volume).
This corresponds in an array with size `(5,4,3)` to 
the indices `(3,2.5,2)`.
"""
struct CenterMiddle <: Center end


"""
    CenterFT

Type to indicate that the center should be 
at the center defined in the FFT sense.
This corresponds in an array with size `(5,4,3)` to 
the indices `(3,3,2)`.
"""
struct CenterFT <: Center end


"""
    center(sz::NTuple{N, T}, ::Type{<:Center})

Return the corresponding center of an array with size `sz`.
Depending on the `Center` type the center is chosen.
See [`CenterFirst`](@ref), [`CenterLast`](@ref),
[`CenterMiddle`](@ref), [`CenterFT`](@ref).

```jldoctest
julia> center((1,2,3,4), CenterFirst)
(1, 1, 1, 1)

julia> center((1,2,3,4), CenterLast)
(1, 2, 3, 4)

julia> center((1,2,3,4), CenterMiddle)
(1, 1.5, 2, 2.5)

julia> center((1,2,3,4), CenterFT)
(1, 2, 2, 3)
```
"""
center(size::NTuple{N, T}, 
       ::Type{CenterFirst}) where {T, N} = ntuple(i -> one(T), Val(N))

center(size::NTuple{N, T}, 
       ::Type{CenterLast}) where {T, N} = ntuple(i -> size[i], Val(N))

center(size::NTuple{N, T}, 
       ::Type{CenterMiddle}) where {T, N} = ntuple(i -> (1 + size[i]) / 2, Val(N))

center(size::NTuple{N, T}, 
       ::Type{CenterFT}) where {T, N} = ntuple(i -> size[i] ÷ 2 + 1, Val(N))


"""
    center(sz::NTuple{N, T}, j<:Number)

Return an tuple with the same length as `sz` and 
with entries `j`.

 # Examples
```jldoctest
julia> center((2,2), 1)
(1, 1)

julia> center((2,2, 5), 3)
(3, 3, 3)
```
"""
center(size::NTuple{N, T}, 
       j::Number) where {T, N} = ntuple(i -> j, Val(N))

"""
    center(sz::NTuple{N, T}, ctr::NTuple{N, T})

Return ctr.

 # Examples
```jldoctest
julia> center((1,2), (1,1))
(1, 1)

julia> center((1,3), (1,1))
(1, 1)
```
"""
center(size::NTuple{N, T}, 
       ctr::NTuple{N, T}) where {T, N} = ctr 
