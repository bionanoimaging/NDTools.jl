## MutablePaddedViews.jl

struct MutablePaddedView{T,N,I,A} <: AbstractArray{T,N}
    data::PaddedView{T,N,I,A}
end

Base.size(A::MutablePaddedView) = size(A.data)
# Base.ndims(A::MutablePaddedView) = ndims(A.data)

# The change below makes select_region into a writable region
Base.@propagate_inbounds function Base.setindex!(A::MutablePaddedView{T,N}, v, pos::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A.data, pos...)
    if Base.checkbounds(Bool, A.data.data, pos...)
        return setindex!(A.data.data, v, pos... )
    else
        return A.data.fillvalue
    end
end

Base.@propagate_inbounds function Base.getindex(A::MutablePaddedView{T,N}, pos::Vararg{Int,N}) where {T,N}
    getindex(A.data, pos...)
end
