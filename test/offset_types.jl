@testset "Offset Types" begin
    @test center((1,2), (1,1)) == (1,1)
    @test center((100,2), (5,1)) == (5,1)
    

    @test center((1,2), 2) == (2,2)
    @test center((100,2,2,1), 5) == (5,5,5,5)

end
