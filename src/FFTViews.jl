module FFTViews

using Base: tail, unsafe_length

using CustomUnitRanges
include(CustomUnitRanges.filename_for_urange)

# export FFTPhysView, FFTFreqView
export FFTFilterView

abstract AbstractFFTView{T,N} <: AbstractArray{T,N}

for V in (:FFTFilterView,) # PhysView, :FFTFreqView)
    @eval begin
        immutable $V{T,N,A<:AbstractArray} <: AbstractFFTView{T,N}
            parent::A

            function $V(parent::A)
                all(x->first(x)==1, indices(parent)) || throw(ArgumentError("indices of parent must start with 1"))
                new(parent)
            end
        end
        $V{T,N}(parent::AbstractArray{T,N}) = $V{T,N,typeof(parent)}(parent)
        (::Type{$V{T,N}}){T,N}(dims::Dims{N}) = $V(Array{T,N}(dims))
        (::Type{$V{T}}  ){T,N}(dims::Dims{N}) = $V(Array{T,N}(dims))

        @inline function Base.getindex{T,N}(F::$V{T,N}, I::Vararg{Int,N})
            inds = indices(F)
            Base.checkbounds_indices(Bool, inds, I) || Base.throw_boundserror(F, I)
            F.parent[reindex($V, inds, I)...]
        end

        @inline function Base.setindex!{T,N}(F::$V{T,N}, val, I::Vararg{Int,N})
            inds = indices(F)
            Base.checkbounds_indices(Bool, inds, I) || Base.throw_boundserror(F, I)
            F.parent[reindex($V, inds, I)...] = val
        end

    end
end

Base.parent(F::AbstractFFTView) = F.parent
Base.indices(F::AbstractFFTView) = map(indrange, indices(parent(F)))
indrange(i) = (h = last(i)>>1; URange(-h, -h+last(i)-1))

Base.fft(F::AbstractFFTView) = fft(parent(F))

@inline reindex{V}(::Type{V}, inds, I) = (_reindex(V, inds[1], I[1]), reindex(V, tail(inds), tail(I))...)
reindex{V}(::Type{V}, ::Tuple{}, ::Tuple{}) = ()
# _reindex(::Type{FFTPhysView}, ind, i) = i-first(ind)+1
# _reindex(::Type{FFTFreqView}, ind, i) = i+last(ind)+1
_reindex(::Type{FFTFilterView}, ind, i) = ifelse(i < 0, i+unsafe_length(ind)+1, i+1)

end # module
