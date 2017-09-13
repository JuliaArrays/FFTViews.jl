using FFTViews
using Base.Test

function test_approx_eq_periodic(a::FFTView, b)
    for I in CartesianRange(indices(b))
        @test a[I-1] ≈ b[I]
    end
    nothing
end

function test_approx_eq_periodic(a::FFTView, b::FFTView)
    for I in CartesianRange(indices(b))
        @test a[I] ≈ b[I]
    end
    nothing
end


@testset "basics" begin
    a = FFTView{Float64,2}((5,7))
    @test indices(a) == (0:4, 0:6)
    @test eltype(a) == Float64
    a = FFTView{Float64}((5,7))
    @test indices(a) == (0:4, 0:6)
    @test eltype(a) == Float64
    @test_throws MethodError FFTView{Float64,3}((5,7))
    for i = 1:35
        a[i] = i
    end
    @test a[3,1] == 9
    @test a[:,0] == FFTView(collect(1:5))
    @test a[0,:] == FFTView(collect(1:5:35))
    @test a[1,0:7] == [2:5:35;2]
    @test a[2,[0,1,0,-1]] == [3,8,3,33]
    @test a[3,trues(9)] == [9,14,19,24,29,34,4,9,14]
    @test a[3,FFTView(trues(9))] == [4,9,14,19,24,29,34,4,9]
    b = similar(Array{Int}, indices(a))
    @test isa(b, FFTView)
    @test indices(b) == indices(a)
    @test eltype(b) == Int
    @test reshape(a, Val{2}) === a
    @test reshape(a, Val{1}) == FFTView(convert(Vector{Float64}, collect(1:35)))
    @test indices(reshape(a, Val{3})) == (0:4,0:6,0:0)
end

@testset "convolution-shift" begin
    for l in (8,9)
        a = zeros(l)
        v = FFTView(a)
        @test indices(v,1) == 0:l-1
        v[0] = 1
        p = rand(l)
        pfilt = ifft(fft(p).*fft(v))
        @test real(pfilt) ≈ p
        v[0] = 0
        v[-1] = 1
        pfilt = ifft(fft(p).*fft(v))
        @test real(pfilt) ≈ circshift(p, -1)
        v[-1] = 0
        v[+1] = 1
        pfilt = ifft(fft(p).*fft(v))
        @test real(pfilt) ≈ circshift(p, +1)
    end
    for l2 in (8,9), l1 in (8,9)
        a = zeros(l1,l2)
        v = FFTView(a)
        @test indices(v) == (0:l1-1, 0:l2-1)
        p = rand(l1,l2)
        for offset in ((0,0), (-1,0), (0,-1), (-1,-1),
                       (1,0), (0,1), (1,1), (1,-1), (-1,1),
                       (3,-5), (281,-14))
            fill!(a, 0)
            v[offset...] = 1
            pfilt = ifft(fft(p).*fft(v))
            @test real(pfilt) ≈ circshift(p, offset)
        end
    end
end

using OffsetArrays

@testset "convolution-offset" begin
    for l2 in (8,9), l1 in (8,9)
        a = OffsetArray(zeros(l1,l2), (-2,-3))
        v = FFTView(a)
        @test indices(v) == (-2:l1-3, -3:l2-4)
        p = rand(l1,l2)
        po = OffsetArray(copy(p), (5,-1))
        for offset in ((0,0), (-1,0), (0,-1), (-1,-1),
                       (1,0), (0,1), (1,1), (1,-1), (-1,1),
                       (3,-5), (281,-14))
            fill!(a, 0)
            v[offset...] = 1
            pfilt = ifft(fft(p).*fft(v))
            @test real(pfilt) ≈ circshift(p, offset)
            pofilt = ifft(fft(po).*fft(v))
            test_approx_eq_periodic(FFTView(real(pofilt)), circshift(po, offset))
            pfilt = irfft(rfft(p).*rfft(v), length(indices(v,1)))
            @test real(pfilt) ≈ circshift(p, offset)
            pofilt = irfft(rfft(po).*rfft(v), length(indices(v,1)))
            test_approx_eq_periodic(FFTView(real(pofilt)), circshift(po, offset))
            dims = (1,2)
            pfilt = ifft(fft(p, dims).*fft(v, dims), dims)
            @test real(pfilt) ≈ circshift(p, offset)
            pofilt = ifft(fft(po, dims).*fft(v, dims), dims)
            test_approx_eq_periodic(FFTView(real(pofilt)), circshift(po, offset))
            pfilt = irfft(rfft(p, dims).*rfft(v, dims), length(indices(v,1)), dims)
            @test real(pfilt) ≈ circshift(p, offset)
            pofilt = irfft(rfft(po, dims).*rfft(v, dims), length(indices(v,1)), dims)
            test_approx_eq_periodic(FFTView(real(pofilt)), circshift(po, offset))
        end
    end
end

@testset "vector indexing" begin
    v = FFTView(1:10)
    @test v[-10:15] == [1:10;1:10;1:6]
end

nothing
