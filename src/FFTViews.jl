module FFTViews

using Base: tail, unsafe_length

# A custom rangetype that will be used for indices and never throws a
# boundserror because the domain is actually periodic.
using CustomUnitRanges
include(CustomUnitRanges.filename_for_urange)

Base.checkindex(::Type{Bool}, inds::URange, ::Colon) = true
Base.checkindex(::Type{Bool}, inds::URange, ::Real) = true
Base.checkindex(::Type{Bool}, inds::URange, ::Range) = true
Base.checkindex(::Type{Bool}, inds::URange, ::AbstractArray) = true
Base.checkindex(::Type{Bool}, inds::URange, ::AbstractArray{Bool}) = true

export FFTView

abstract AbstractFFTView{T,N} <: AbstractArray{T,N}

for V in (:FFTView,) #(:FFTFilterView,) # PhysView, :FFTFreqView)
    @eval begin
        immutable $V{T,N,A<:AbstractArray} <: AbstractFFTView{T,N}
            parent::A

            function $V(parent::A)
                new(parent)
            end
        end
        $V{T,N}(parent::AbstractArray{T,N}) = $V{T,N,typeof(parent)}(parent)
        (::Type{$V{T,N}}){T,N}(dims::Dims{N}) = $V(Array{T,N}(dims))
        (::Type{$V{T}}  ){T,N}(dims::Dims{N}) = $V(Array{T,N}(dims))

        # Note: there are no bounds checks because it's all periodic
        @inline function Base.getindex{T,N}(F::$V{T,N}, I::Vararg{Int,N})
            P = parent(F)
            @inbounds ret = P[reindex($V, indices(P), I)...]
            ret
        end

        @inline function Base.setindex!{T,N}(F::$V{T,N}, val, I::Vararg{Int,N})
            P = parent(F)
            @inbounds P[reindex($V, indices(P), I)...] = val
        end

    end
end

Base.parent(F::AbstractFFTView) = F.parent
Base.indices(F::AbstractFFTView) = map(indrange, indices(parent(F)))
indrange(i) = URange(first(i)-1, last(i)-1)

Base.fft(F::FFTView; kwargs...) = fft(parent(F); kwargs...)
Base.rfft(F::FFTView; kwargs...) = rfft(parent(F); kwargs...)
Base.fft(F::FFTView, dims; kwargs...) = fft(parent(F), dims; kwargs...)
Base.rfft(F::FFTView, dims; kwargs...) = rfft(parent(F), dims; kwargs...)

@inline reindex{V}(::Type{V}, inds, I) = (_reindex(V, inds[1], I[1]), reindex(V, tail(inds), tail(I))...)
reindex{V}(::Type{V}, ::Tuple{}, ::Tuple{}) = ()
# _reindex(::Type{FFTPhysView}, ind, i) = i-first(ind)+1
# _reindex(::Type{FFTFreqView}, ind, i) = i+last(ind)+1
# _reindex(::Type{FFTFilterView}, ind, i) = ifelse(i < 0, i+unsafe_length(ind)+1, i+1)
_reindex(::Type{FFTView}, ind, i) = modrange(i+1, ind)

modrange(i, rng::AbstractUnitRange) = mod(i-first(rng), unsafe_length(rng))+first(rng)

end # module
