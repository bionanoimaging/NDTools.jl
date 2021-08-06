@testset "Test MutablePaddedView" begin
    v = NDTools.MutablePaddedView(PaddedView(22.2, rand(5,6,10), (10,10,10), (3,4,5)));
    @test v[1,1,1] == 22.2
    v[5,5,5] = 10.0
    @test v[5,5,5] == 10.0
    v[4,4,4] = 10.0
    @test v[4,4,4] == 22.2
    v[5:6,5:10,5:10] .= 11.0
    @test v[6,7,7] == 11.0
end

