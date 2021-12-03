@testset "Generation tools" begin

    t = (1,2)
    @test ϕ_tuple(t::NTuple) ≈ atan(2, 1)
    t = (1.23,129.23)
    @test ϕ_tuple(t::NTuple) ≈ atan(t[2], t[1])

    @test idx_to_dim([(1, 2) (3, 4); (5, 6) (7, 8)]) == [1 3; 5 7;;; 2 4; 6 8]

    @test idx_to_dim([(1, 2), (3, 4)]) == [1 2; 3 4]


end
