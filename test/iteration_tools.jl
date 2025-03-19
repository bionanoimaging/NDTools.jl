@testset "Test linear_index" begin
    a = rand(10,10,10);
    @test a[3,4,5]  == a[433]
    @test linear_index((3,4,5), (10,10,10)) == 433
end
@testset "Test cast_iter" begin
    i1 = NDTools.cast_iter([1;2;3;4])
    i2 = NDTools.cast_iter([1,2,3,4])
    i3 = NDTools.cast_iter([(1,),(2,),(3,),(4,)])
    for (i,j,k) in zip(i1,i2,i3)
        @test i == j
        @test j == k[1]
    end
end

@testset "Test cast_number_iter" begin
    i1 = NDTools.cast_number_iter([1;2;3;4])
    i2 = NDTools.cast_number_iter([1,2,3,4])
    i3 = NDTools.cast_number_iter((1,2,3,4))
    for (i,j,k) in zip(i1,i2,i3)
        @test i == j
        @test j == k
    end
end

@testset "Test optional_mat_to_iter" begin
    @test all(NDTools.optional_mat_to_iter([1;2;3;4]) .== NDTools.cast_iter([1;2;3;4]))
end

@testset "Test mat_to_tvec" begin
    @test NDTools.mat_to_tvec([1 3 5; 2 4 6], Val(2)) == [(1,2),(3,4),(5,6)]
end

@testset "Test apply_dims" begin
    @test NDTools.apply_dims((1,2,3),(1,2), 4)  == (1, 2, 0, 0)
    @test NDTools.apply_dims([(1,2,3),(1,2,3),(1,2,3)],(1,2), 4)  == ((1, 2, 0, 0),(1, 2, 0, 0),(1, 2, 0, 0))
end


