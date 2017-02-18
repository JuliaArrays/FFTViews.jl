module FFTViews

using Base: tail, unsafe_length
using Compat

# A custom rangetype that will be used for indices and never throws a
# boundserror because the domain is actually periodic.
using CustomUnitRanges
include(CustomUnitRanges.filename_for_urange)

Base.checkindex(::Type{Bool}, inds::URange, ::Colon) = true
Base.checkindex(::Type{Bool}, inds::URange, ::Real) = true
Base.checkindex(::Type{Bool}, inds::URange, ::Range) = true
Base.checkindex(::Type{Bool}, inds::URange, ::AbstractVector{Bool}) = true
Base.checkindex(::Type{Bool}, inds::URange, ::AbstractArray{Bool}) = true
Base.checkindex(::Type{Bool}, inds::URange, ::AbstractArray) = true

export FFTView

@compat abstract type AbstractFFTView{T,N} <: AbstractArray{T,N} end

immutable FFTView{T,N,A<:AbstractArray} <: AbstractFFTView{T,N}
    parent::A

    function (::Type{FFTView{T,N,A}}){T,N,A}(parent::A)
        new{T,N,A}(parent)
    end
end
FFTView{T,N}(parent::AbstractArray{T,N}) = FFTView{T,N,typeof(parent)}(parent)
(::Type{FFTView{T,N}}){T,N}(dims::Dims{N}) = FFTView(Array{T,N}(dims))
(::Type{FFTView{T}}  ){T,N}(dims::Dims{N}) = FFTView(Array{T,N}(dims))

# Note: there are no bounds checks because it's all periodic
@inline function Base.getindex{T,N}(F::FFTView{T,N}, I::Vararg{Int,N})
    P = parent(F)
    @inbounds ret = P[reindex(FFTView, indices(P), I)...]
    ret
end

@inline function Base.setindex!{T,N}(F::FFTView{T,N}, val, I::Vararg{Int,N})
    P = parent(F)
    @inbounds P[reindex(FFTView, indices(P), I)...] = val
end

Base.parent(F::AbstractFFTView) = F.parent
Base.indices(F::AbstractFFTView) = map(indrange, indices(parent(F)))
indrange(i) = URange(first(i)-1, last(i)-1)

function Base.similar(A::AbstractArray, T::Type, shape::Tuple{URange,Vararg{URange}})
    all(x->first(x)==0, shape) || throw(BoundsError("cannot allocate FFTView with the first element of the range non-zero"))
    FFTView(similar(A, T, map(length, shape)))
end

function Base.similar(f::Union{Function,DataType}, shape::Tuple{URange,Vararg{URange}})
    all(x->first(x)==0, shape) || throw(BoundsError("cannot allocate FFTView with the first element of the range non-zero"))
    FFTView(similar(f, map(length, shape)))
end

Base.reshape{_,N}(F::FFTView{_,N}, ::Type{Val{N}})   = F
Base.reshape{_,M,N}(F::FFTView{_,M}, ::Type{Val{N}}) = FFTView(reshape(parent(F), Val{N}))

Base.fft(F::FFTView; kwargs...) = fft(parent(F); kwargs...)
Base.rfft(F::FFTView; kwargs...) = rfft(parent(F); kwargs...)
Base.fft(F::FFTView, dims; kwargs...) = fft(parent(F), dims; kwargs...)
Base.rfft(F::FFTView, dims; kwargs...) = rfft(parent(F), dims; kwargs...)

@inline reindex{V}(::Type{V}, inds, I) = (_reindex(V, inds[1], I[1]), reindex(V, tail(inds), tail(I))...)
reindex{V}(::Type{V}, ::Tuple{}, ::Tuple{}) = ()
_reindex(::Type{FFTView}, ind, i) = modrange(i+1, ind)

modrange(i, rng::AbstractUnitRange) = mod(i-first(rng), unsafe_length(rng))+first(rng)

end # module
