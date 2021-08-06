@testset "Test soft_theta" begin 
    @test soft_theta(0.01) == 1.0
    @test soft_theta(-0.01) == 0.0
    @test soft_theta(-0.0) ≈ 0.5
    @test soft_theta(1/3.0, 1.0) ≈ 0.75
end

@testset "Test soft_delta" begin 
    @test soft_delta(0.01) == 0.0
    @test soft_delta(-0.01) == 0.0
    @test soft_delta(-0.0) ≈ 1.0
    @test soft_delta(1/3.0, 1.0) ≈ 0.75
end

