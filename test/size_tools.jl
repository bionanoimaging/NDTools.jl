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
    @test (1, 3, 2) == select_sizes(randn((4,3,2)), (2,3))
    @test (3, 2) == select_sizes(randn((4,3,2)), (2,3), keep_dims=false)
    @test (1, ) == select_sizes(randn((1,)), (1,), keep_dims=false)
end

@testset "Test single_dim_size" begin
    @test single_dim_size(4, 3) == (1, 1, 1, 3)        
    @test single_dim_size(4, 5) == (1, 1, 1, 5)
    @test single_dim_size(2, 5) == (1, 5)
end

@testset "Test reorient" begin
    @test size(reorient([1,2,3,4], 3)) == (1,1,4)
    @test reorient([1,2,3,4], 2)[:] == [1,2,3,4]
end

@testset "Test collect_dim" begin
    @test (1,1,1,5) == size(collect_dim((3,4,5,6,7),4))
    @test collect_dim((3,4,5,6,7),4)[1,1,1,3] == 5
    @test collect_dim(1:5,4)[:] == [1,2,3,4,5]
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

@testset "Test optional_posZ" begin
    @test NDTools.optional_posZ((1,2,3,4), (1,1,1,1)) == 2
    @test NDTools.optional_posZ((5,5), (1,1)) == 1
end

@testset "Test curry" begin
    g = NDTools.curry(+,10.0)
    @test g(3) == 13
end


