
    @testset "Test IterType" begin
        # const IterType = Union{NTuple{N,Tuple} where N, Vector, Matrix, Base.Iterators.Repeated}
        @test isa([(1,2),(1,2)], NDTools.IterType)
        @test isa(((1,2),(1,2)), NDTools.IterType)
        @test isa([1,2,3,4], NDTools.IterType)
        @test isa([1 2;3 4], NDTools.IterType)
    end


    @testset "Test default_type" begin
        @test NDTools.default_type(Int32, Float32) == Float32
        @test NDTools.default_type(Float64, Float32) == Float64
        @test NDTools.default_type(ComplexF64, Float32) == ComplexF64 
        @test NDTools.default_type(Float32, Float64) == Float32  # Should this really be like this?
    end

    @testset "Test default_type" begin
        @test apply_tuple_list(+, 1, 2) == 3 
        @test apply_tuple_list(+, [1,2,3], 2) == (3,4,5) 
        @test apply_tuple_list(.+, 1, [(1,2),(3,4)]) == ((2, 3), (4, 5))
    end

    @testset "Test get_complex_datatype" begin
        @test get_complex_datatype(22.2) == ComplexF64
        @test get_complex_datatype(22.2f0) == ComplexF32
        @test get_complex_datatype(22) == Complex{Int64}
    end

