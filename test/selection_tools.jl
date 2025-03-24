
    @testset "Test slice" begin    
        x = opt_cu(randn((1,2,3,4)))
        y = NDTools.slice(x, 2, 2)
        @test x[:, 2:2, :, :] == y

        x = opt_cu(randn((5,2,3,4)))
        y = NDTools.slice(x, 1, 4)
        @test x[4:4, :, :, :] == y
        y = NDTools.slice(x, 1, 2:3)
        @test x[2:3, :, :, :] == y
        y = NDTools.slice(x, 4, 4:4)
        @test x[:, :, :, 4:4] == y

        x = opt_cu(randn((5)))
        y = NDTools.slice(x, 1, 5)
        @test x[5:5] == y

    end
    
    
    @testset "Test slice indices" begin
        x = opt_cu(randn((1,2,3)))
        y = NDTools.slice_indices(axes(x), 1, 1)
        @test y == (1:1, 1:2, 1:3)
        
        x = opt_cu(randn((20,4,20, 1, 2)))
        y = NDTools.slice_indices(axes(x), 2, 3)
        @test y == (1:20, 3:3, 1:20, 1:1, 1:2)
    end


    @testset "Flatten trailing dims" begin
        x = opt_cu(ones((1, 1, 2, 2)))
        @test flatten_trailing_dims(x, Val(1)) == opt_cu([1.0, 1.0, 1.0, 1.0])
        @test flatten_trailing_dims(x, Val(2)) == opt_cu([1.0 1.0 1.0 1.0])
        @test flatten_trailing_dims(x, Val(3)) == opt_cu([1.0;;; 1.0;;; 1.0;;; 1.0])
        @test flatten_trailing_dims(x, Val(4)) == opt_cu([1.0;;; 1.0;;;; 1.0;;; 1.0])
    end

    @testset "Test expand_dims second" begin
        x = opt_cu(zeros(Int, 2,2,2,2,2,2))
        @test size(expand_dims(x,1,2,3)) == opt_cu((1,1,1,2,2,2,2,2,2))
        @test size(expand_dims(x,3)) == opt_cu((2,2,1,2,2,2,2))
        @test size(expand_dims(x,6)) == opt_cu((2,2,2,2,2,1, 2))
        @test size(expand_dims(x,7)) == opt_cu((2,2,2,2,2,2,1))
        @test size(expand_dims(x,1)) == opt_cu((1, 2,2,2,2,2,2))
        x = opt_cu(zeros(Int, 2))
        @test size(expand_dims(x,1)) == (1,2)
        @test size(expand_dims(x,2)) == (2,1)

        @test expand_dims(zeros((2,2)), 1) |> size == (1, 2, 2)
        @test expand_dims(zeros((2,2)), 2) |> size == (2, 1, 2)
        @test expand_dims(zeros((2,2)), 3) |> size == (2, 2, 1)
        @test expand_dims(zeros((2,2)), 1,3,4) |> size == (1, 2, 1, 1, 2)
    end
    
    @testset "Test expand_dims" begin
        function f(s, N)
            @test expand_dims(randn(s), Val(N + length(s)))|> size == (s..., ones(Int,N)...)
        end
        f(opt_cu((1,2,3)), 2)
        f(opt_cu((1,2,3,4,5)), 8)
        f(opt_cu((1)), 5)
    end

    @testset "Test select_region" begin
        x = opt_cu(ones(3,))
        @test select_region_view(x; new_size=(7,),center=(1,), pad_value=-1) == opt_cu([-1.0;-1.0;-1.0;1.0;1.0;1.0;-1.0])
        @test select_region_view(x, new_size=(7,); center=(1,), pad_value=-1) == opt_cu([-1.0;-1.0;-1.0;1.0;1.0;1.0;-1.0]) # test the alias
        for d=1:5
            for n=1:10
                sz = Tuple(rand(1:5) for q in 1:d)
                a = rand(sz...)
                nsz = Tuple(rand(1:5) for q in 1:d)
                nc = Tuple(rand(1:5) for q in 1:d)
                pad = rand()
                @test select_region(a,new_size=nsz, center=nc, pad_value=pad) == select_region(a,new_size=nsz, center=nc, pad_value=pad)
                @test select_region(a, nsz, center=nc, pad_value=pad) == select_region(a,new_size=nsz, center=nc, pad_value=pad)
                @test select_region(a, nsz; center=nc, pad_value=pad) == select_region(a,new_size=nsz, center=nc, pad_value=pad)
            end
        end
        a = opt_cu(ones(10,10))
        @test all(select_region(a) .== 1) # simplest version
        nz = (20,20)
        @test all(select_region(a, new_size=nz, pad_value=1) .== 1) # with padding
        @test all(select_region(a, nz; pad_value=1) .== 1) # test the alias
        @test all(select_region(a, new_size=nz, center=(-100,100), pad_value=10) .== 10) # only pad values
        @test all(select_region(a, nz; center=(-100,100), pad_value=10) .== 10) # test the alias
        function f(a,b) a.+=1 end # user-defined function
        @test all(select_region!(2 .*a, a, operator! = f) .== 2) # let the operator add one to destination
        @test select_region(collect(1:10), new_size=(5,), center=(1,), dst_center=(1,)) == collect(1:5)
        @test select_region_view(collect(1:10), (5,), center=(1,), dst_center=(1,)) == collect(1:5)
        @test select_region_view(collect(1:10), new_size=(5,), center=(1,), dst_center=(1,)) == collect(1:5)
        @test select_region_view(collect(1:10), (5,); center=(1,), dst_center=(1,)) == collect(1:5) # test the alias

        a = opt_cu(ones(10,10))
        select_region!(2 .*a, a, dst_center=(10,10));
        @test all(a[5:end,5:end] .== 2) # check the overwritten part
        @test all(a[1:4,1:4] .== 1) # check part of the non-overwritten part

        a = opt_cu(ones(10,10))
        b = 4*opt_cu(ones(70,70));
        select_region!(b, a, size(a), dst_center=(-20,20));
        select_region!(b, a, dst_center=(-20, 20));
        @test all(a .== 4) # check the automatic selection of a large enough new_size
    end

    @testset "Test Magnificiation" begin
        x = opt_cu([1, 2, 3])
        @test select_region(x, M = 2) == opt_cu([0, 0, 1, 2, 3, 0])

        @test select_region(x, M = 0.5) == opt_cu([1, 2])
        @test select_region(x, M = 0.49) == opt_cu([2])
        @test select_region(x, M = 2.4) == opt_cu([0, 0, 1, 2, 3, 0, 0])
        y = opt_cu([1 2; 3 4]);
        @test select_region(y, M = 2) == opt_cu([0 0 0 0; 0 1 2 0; 0 3 4 0; 0 0 0 0])
        @test_throws AssertionError select_region(opt_cu([1]), M = 1, new_size=(2,1))
        @test_throws AssertionError select_region(opt_cu([1]), (2,1), M = 1) # test the alias
    end


    @testset "Test assignment functions" begin
        dst = opt_cu(ones(10,10))
        src = opt_cu(ones(10,10))
        @test all(NDTools.assign_to!(dst, src) .== 1)
        dst .= 1
        @test all(NDTools.add_to!(dst, src) .== 2)
        dst .= 1
        @test all(NDTools.sub_to!(dst, src) .== 0)
        src .= 2
        dst .= 2
        @test all(NDTools.mul_to!(dst, src) .== 4)
        src .= 2
        dst .= 1
        @test all(NDTools.div_to!(dst, src) .== 0.5)
    end

