using PaddedViews, IndexFunArrays
Random.seed!(42)

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

    @testset "Test slice" begin
    
        x = randn((1,2,3,4))
        y = NDTools.slice(x, 2, 2)
        @test x[:, 2:2, :, :] == y

        x = randn((5,2,3,4))
        y = NDTools.slice(x, 1, 4)
        @test x[4:4, :, :, :] == y

        x = randn((5))
        y = NDTools.slice(x, 1, 5)
        @test x[5:5] == y

    end
    
    
    @testset "Test slice indices" begin
        x = randn((1,2,3))
        y = NDTools.slice_indices(axes(x), 1, 1)
        @test y == (1:1, 1:2, 1:3)
    
    
        x = randn((20,4,20, 1, 2))
        y = NDTools.slice_indices(axes(x), 2, 3)
        @test y == (1:20, 3:3, 1:20, 1:1, 1:2)
    end


    @testset "Test expand_dims" begin
        function f(s, N)
            @test expand_dims(randn(s), N + length(s))|> size == (s..., ones(Int,N)...)
            @test expand_dims(randn(s), Val(N + length(s)))|> size == (s..., ones(Int,N)...)
        end
        f((1,2,3), 2)
        f((1,2,3,4,5), 8)
        f((1), 5)
    end

    @testset "Test collect_dim" begin
        @test (1,1,1,5) == size(collect_dim((3,4,5,6,7),4))
        @test collect_dim((3,4,5,6,7),4)[1,1,1,3] == 5
        @test collect_dim(1:5,4)[:] == [1,2,3,4,5]
    end

    @testset "Test linear_index" begin
        a = rand(10,10,10);
        @test a[3,4,5]  == a[433]
        @test linear_index((3,4,5), (10,10,10)) == 433
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

    @testset "Test expand_size" begin
        @test (1,2,3,7,8) == Tuple(expand_size((1,2,3), (4,5,6,7,8)))
        @test (1,2) == Tuple(expand_size((1,2,3,4,5,6), (4,5)))
    end

    @testset "Test expand_add" begin
        @test expand_add((1,2,3),(4,5,6,7,8,9)) == (5, 7, 9, 7, 8, 9)
    end

    @testset "Test optional_posZ" begin
        @test NDTools.optional_posZ((1,2,3,4), (1,1,1,1)) == 2
        @test NDTools.optional_posZ((5,5), (1,1)) == 1
    end

    @testset "Test curry" begin
        g = NDTools.curry(+,10.0)
        @test g(3) == 13
    end

    @testset "Test IterType" begin
        # const IterType = Union{NTuple{N,Tuple} where N, Vector, Matrix, Base.Iterators.Repeated}
        @test isa([(1,2),(1,2)], NDTools.IterType)
        @test isa(((1,2),(1,2)), NDTools.IterType)
        @test isa([1,2,3,4], NDTools.IterType)
        @test isa([1 2;3 4], NDTools.IterType)
    end
    @testset "Test get_complex_datatype" begin
        @test get_complex_datatype(22.2) == ComplexF64
        @test get_complex_datatype(22.2f0) == ComplexF32
        @test get_complex_datatype(22) == Complex{Int64}
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
        @test NDTools.mat_to_tvec([1 3 5; 2 4 6]) == [(1,2),(3,4),(5,6)]
    end

    @testset "Test apply_dims" begin
        @test NDTools.apply_dims((1,2,3),(1,2), 4)  == (1, 2, 0, 0)
        @test NDTools.apply_dims([(1,2,3),(1,2,3),(1,2,3)],(1,2), 4)  == ((1, 2, 0, 0),(1, 2, 0, 0),(1, 2, 0, 0))
    end


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

    @testset "Test select_region" begin
        @test select_region(ones(3,),new_size=(7,),center=(1,), pad_value=-1) == [-1.0;-1.0;-1.0;1.0;1.0;1.0;-1.0]
        for d=1:5
            for n=1:10
                sz = Tuple(rand(1:5) for q in 1:d)
                a = rand(sz...)
                nsz = Tuple(rand(1:5) for q in 1:d)
                nc = Tuple(rand(1:5) for q in 1:d)
                pad = rand()
                @test all(select_region(a,new_size=nsz, center=nc, pad_value=pad) .== select_region!(a,new_size=nsz, center=nc, pad_value=pad))
            end
        end
        a = ones(10,10)
        @test all(select_region!(a) .== 1) # simplest version
        nz = (20,20)
        @test all(select_region!(a, new_size=nz, pad_value=1) .== 1) # with padding
        @test all(select_region!(a, new_size=nz, center=(-100,100), pad_value=10) .== 10) # only pad values
        function f(a,b) a.+=1 end # user-defined function
        @test all(select_region!(2 .*a, a, operator! = f) .== 2) # let the operator add one to destination
    end

    @testset "Test assignment functions" begin
        dst = ones(10,10)
        src = ones(10,10)
        @test all(assign_to!(dst,src) .== 1)
        dst = ones(10,10)
        @test all(add_to!(dst,src) .== 2)
        dst = ones(10,10)
        @test all(sub_to!(dst,src) .== 0)
        src = ones(10,10) .* 2
        dst = ones(10,10) .* 2
        @test all(mul_to!(dst,src) .== 4)
        src = ones(10,10) .* 2
        dst = ones(10,10)
        @test all(div_to!(dst,src) .== 0.5)
    end

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

    @testset "Test radial_mean" begin 
        a = rr((10,10)) .< 5.0
        (rm, ax) = radial_mean(a)
        @test rm[1] ≈ 1.0
        @test rm[5] ≈ 1.0
        @test rm[6] ≈ 0.0
        @test ax[1] == 0.5
        @test ax[end] == 7.5
    end
    @testset "Test radial_mean" begin 
        @test Δ_phase([1,1im,-1,-1im,1], 1) ≈ -pi/2 .* ones(4)
    end
