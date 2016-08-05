using FFTViews
using Base.Test

@testset "convolution-shift" begin
    for l in (8,9)
        a = zeros(l)
        v = FFTFilterView(a)
        v[0] = 1   # this should not translate a test pattern
        p = rand(l)
        pfilt = ifft(fft(p).*fft(a))
        @test_approx_eq real(pfilt) p
    end
end
