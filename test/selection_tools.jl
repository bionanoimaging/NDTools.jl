
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


    @testset "Flatten trailing dims" begin
        @test flatten_trailing_dims(ones((1, 1, 2, 2)), Val(1)) == [1.0, 1.0, 1.0, 1.0]
        @test flatten_trailing_dims(ones((1, 1, 2, 2)), Val(2)) == [1.0 1.0 1.0 1.0]
        @test flatten_trailing_dims(ones((1, 1, 2, 2)), Val(3)) == [1.0;;; 1.0;;; 1.0;;; 1.0]
        @test flatten_trailing_dims(ones((1, 1, 2, 2)), Val(4)) == [1.0;;; 1.0;;;; 1.0;;; 1.0]
    end

    
    @testset "Test expand_dims" begin
        function f(s, N)
            @test expand_dims(randn(s), Val(N + length(s)))|> size == (s..., ones(Int,N)...)
        end
        f((1,2,3), 2)
        f((1,2,3,4,5), 8)
        f((1), 5)
    end

    @testset "Test select_region" begin
        @test select_region_view(ones(3,),new_size=(7,),center=(1,), pad_value=-1) == [-1.0;-1.0;-1.0;1.0;1.0;1.0;-1.0]
        for d=1:5
            for n=1:10
                sz = Tuple(rand(1:5) for q in 1:d)
                a = rand(sz...)
                nsz = Tuple(rand(1:5) for q in 1:d)
                nc = Tuple(rand(1:5) for q in 1:d)
                pad = rand()
                @test all(select_region(a,new_size=nsz, center=nc, pad_value=pad) .== select_region(a,new_size=nsz, center=nc, pad_value=pad))
            end
        end
        a = ones(10,10)
        @test all(select_region(a) .== 1) # simplest version
        nz = (20,20)
        @test all(select_region(a, new_size=nz, pad_value=1) .== 1) # with padding
        @test all(select_region(a, new_size=nz, center=(-100,100), pad_value=10) .== 10) # only pad values
        function f(a,b) a.+=1 end # user-defined function
        @test all(select_region!(2 .*a, a, operator! = f) .== 2) # let the operator add one to destination
    end

    @testset "Test assignment functions" begin
        dst = ones(10,10)
        src = ones(10,10)
        @test all(NDTools.assign_to!(dst,src) .== 1)
        dst = ones(10,10)
        @test all(NDTools.add_to!(dst,src) .== 2)
        dst = ones(10,10)
        @test all(NDTools.sub_to!(dst,src) .== 0)
        src = ones(10,10) .* 2
        dst = ones(10,10) .* 2
        @test all(NDTools.mul_to!(dst,src) .== 4)
        src = ones(10,10) .* 2
        dst = ones(10,10)
        @test all(NDTools.div_to!(dst,src) .== 0.5)
    end

