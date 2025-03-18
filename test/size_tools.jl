@testset "Test center_position" begin
    @test (2,) == center_position(ones(3))
    @test (3,) == center_position(ones(4))
    @test (3,) == center_position(ones(5))
    @test (2,3,4) == center_position(ones(3,4,6))
end

@testset "Test ft_center_diff" begin
    @test NDTools.ft_center_diff((3,4,5,6)) == (1,2,2,3)
end

@testset "Test center_value" begin
    x = randn((1,2,3,4))
    @test center_value(x) == x[center_position(x)...]
end

@testset "Test select_sizes" begin
    x = opt_cu(randn((4,3,2)))
    @test (1, 3, 2) == select_sizes(x, (2,3))
    @test (1, 3, 1) == select_sizes(x, 2)
    @test (3, 2) == select_sizes_squeeze(x, (2,3))
    y = opt_cu(randn((1,)))
    @test (1, ) == select_sizes_squeeze(randn((1,)), (1,))
    @test (1, ) == select_sizes_squeeze(randn((1,)), 1)
end

@testset "Test single_dim_size" begin
    @test single_dim_size(Val(2), 5) == (1, 5)
    @test single_dim_size(3,5, Val(4)) == (1, 1, 5, 1)
end

@testset "Test reorient" begin
    x = opt_cu([1,2,3,4])
    @test size(reorient(x, Val(3))) == (1,1,4)
    @test size(reorient(x, 3, Val(4))) == (1,1,4,1)
end


@testset "Test expand_size" begin
    @test (1,2,3,7,8) == Tuple(expand_size((1,2,3), (4,5,6,7,8)))
    @test (1,2) == Tuple(expand_size((1,2,3,4,5,6), (4,5)))
end

@testset "Test expand_add" begin
    @test expand_add((1,2,3),(4,5,6,7,8,9)) == (5, 7, 9, 7, 8, 9)
    @test expand_add((1,1), (0,)) == (1,) 
    @test expand_add((1,2), (0,0)) == (1,2) 
    @test expand_add((1,2), (0,0, 0)) == (1,2, 0) 
end

@testset "Test optional_pos_z" begin
    @test NDTools.optional_pos_z((1,2,3,4), (1,1,1,1)) == 2
    @test NDTools.optional_pos_z((5,5), (1,1)) == 1
    @test NDTools.optional_pos_z((5,), (1,)) == 1
end

@testset "Test curry" begin
    g = NDTools.curry(+,10.0)
    @test g(3) == 13
end

@testset "Test axes methods" begin
    @test axes_centered((10,11)) == (-5:4, -5:5)
    @test axes_centered(zeros(10,11)) == (-5:4, -5:5)
    @test axes_corner_only((10,11)) == (-5:0, -5:0)
    @test axes_corner_only(zeros(10,11)) == (-5:0, -5:0)
end


