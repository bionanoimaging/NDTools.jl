@testset "Generation tools" begin

    t = (1,2)
    @test ϕ_tuple(t::NTuple) ≈ atan(2, 1)
    t = (1.23,129.23)
    @test ϕ_tuple(t::NTuple) ≈ atan(t[2], t[1])

    @test idx_to_dim(opt_cu([(1, 2) (3, 4); (5, 6) (7, 8)])) == opt_cu([1 3; 5 7;;; 2 4; 6 8])

    @test idx_to_dim(opt_cu([(1, 2), (3, 4)])) == opt_cu([1 2; 3 4])
    @test NDTools.dim_to_idx(opt_cu([1 2; 3 4])) ==  opt_cu([(1, 2), (3, 4)])

    @test NDTools.dim_to_idx(opt_cu([1 2 2; 3 4 2;;;5 6 2;7 8 2])) == opt_cu([(1, 5)  (2, 6)  (2, 2); (3, 7)  (4, 8)  (2, 2)])

    @test NDTools.idx_to_arr_view(opt_cu([(1, 2) (3, 4); (5, 6) (7, 8)])) == opt_cu([1 5; 2 6;;; 3 7;4 8])

    @test NDTools.idx_to_arr_view(opt_cu([(1, 2) (3, 4); (5, 6) (7, 8)])) == opt_cu([1 5; 2 6;;; 3 7;4 8])
end
