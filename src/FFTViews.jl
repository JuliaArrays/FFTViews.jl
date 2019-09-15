module FFTViews

using Base: tail, unsafe_length, @propagate_inbounds
using FFTW

# A custom rangetype that will be used for indices and never throws a
# boundserror because the domain is actually periodic.
using CustomUnitRanges
include(CustomUnitRanges.filename_for_urange)

@static if isdefined(Base, :IdentityUnitRange)
    const indextypes = (URange, Base.IdentityUnitRange{<:URange})
    const FFTVRange{T} = Union{URange{T}, Base.Slice{URange{T}}, Base.IdentityUnitRange{URange{T}}}
    indrange(i) = Base.IdentityUnitRange(URange(first(i)-1, last(i)-1))
else
    const indextypes = (URange, Base.Slice{<:URange})
    const FFTVRange{T} = Union{URange{T}, Base.Slice{URange{T}}}
    indrange(i) = Base.Slice(URange(first(i)-1, last(i)-1))
end

for T in indextypes
    @eval begin
        Base.checkindex(::Type{Bool}, ::$T, ::Base.Slice) = true
        Base.checkindex(::Type{Bool}, ::$T, ::Base.LogicalIndex) = true
        Base.checkindex(::Type{Bool}, ::$T, ::Real) = true
        Base.checkindex(::Type{Bool}, ::$T, ::AbstractRange) = true
        Base.checkindex(::Type{Bool}, ::$T, ::AbstractVector{Bool}) = true
        Base.checkindex(::Type{Bool}, ::$T, ::AbstractArray{Bool}) = true
        Base.checkindex(::Type{Bool}, ::$T, ::AbstractArray) = true
    end
    if isdefined(Base, :IdentityUnitRange)
        @eval Base.checkindex(::Type{Bool}, ::$T, ::Base.IdentityUnitRange) = true
    end
end

export FFTView

abstract type AbstractFFTView{T,N} <: AbstractArray{T,N} end

struct FFTView{T,N,A<:AbstractArray} <: AbstractFFTView{T,N}
    parent::A

    function FFTView{T,N,A}(parent::A) where {T,N,A}
        new{T,N,A}(parent)
    end
end
FFTView(parent::AbstractArray{T,N}) where {T,N} = FFTView{T,N,typeof(parent)}(parent)
FFTView{T,N}(dims::Dims{N}) where {T,N} = FFTView(Array{T,N}(undef, dims))
FFTView{T}(dims::Dims{N}) where {T,N} = FFTView(Array{T,N}(undef, dims))

# Note: there are no bounds checks because it's all periodic
@inline @propagate_inbounds function Base.getindex(F::FFTView{T,N}, I::Vararg{Int,N}) where {T,N}
    P = parent(F)
    @inbounds ret = P[reindex(FFTView, axes(P), I)...]
    ret
end

@inline @propagate_inbounds function Base.setindex!(F::FFTView{T,N}, val, I::Vararg{Int,N}) where {T,N}
    P = parent(F)
    @inbounds P[reindex(FFTView, axes(P), I)...] = val
end

Base.parent(F::AbstractFFTView) = F.parent
Base.axes(F::AbstractFFTView) = map(indrange, axes(parent(F)))
Base.size(F::AbstractFFTView) = size(parent(F))

function Base.similar(A::AbstractArray, T::Type, shape::Tuple{FFTVRange,Vararg{FFTVRange}})
    all(x->first(x)==0, shape) || throw(BoundsError("cannot allocate FFTView with the first element of the range non-zero"))
    FFTView(similar(A, T, map(length, shape)))
end

function Base.similar(f::Union{Function,Type}, shape::Tuple{FFTVRange,Vararg{FFTVRange}})
    all(x->first(x)==0, shape) || throw(BoundsError("cannot allocate FFTView with the first element of the range non-zero"))
    FFTView(similar(f, map(length, shape)))
end

Base.reshape(F::FFTView{_,N}, ::Type{Val{N}}) where {_,N}   = F
Base.reshape(F::FFTView{_,M}, ::Type{Val{N}}) where {_,M,N} = FFTView(reshape(parent(F), Val(N)))

FFTW.fft(F::FFTView; kwargs...) = fft(parent(F); kwargs...)
FFTW.rfft(F::FFTView; kwargs...) = rfft(parent(F); kwargs...)
FFTW.fft(F::FFTView, dims; kwargs...) = fft(parent(F), dims; kwargs...)
FFTW.rfft(F::FFTView, dims; kwargs...) = rfft(parent(F), dims; kwargs...)

@inline reindex(::Type{V}, inds, I) where {V} = (_reindex(V, inds[1], I[1]), reindex(V, tail(inds), tail(I))...)
reindex(::Type{V}, ::Tuple{}, ::Tuple{}) where {V} = ()
_reindex(::Type{FFTView}, ind, i) = modrange(i+1, ind)

modrange(i, rng::AbstractUnitRange) = mod(i-first(rng), unsafe_length(rng))+first(rng)

end # module
