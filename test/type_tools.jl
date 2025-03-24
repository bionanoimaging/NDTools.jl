@testset "Test Type Tools" begin
    sz = (11,12)
    if use_cuda && CUDA.functional()
        MyA = CuArray
        MyM = CuMatrix
        MyV = CuVector
    else
        MyA = Array
        MyM = Matrix
        MyV = Vector
    end

    # The <: subttype operator is necessary, since the extracted type also contains the storage type information in CUDA
    @test real_arr_type(MyA{Float32, 2}) <: MyM{Float32}
    @test complex_arr_type(MyA{Float32,1}, Val(2)) <: MyM{ComplexF32}
    @test real_arr_type(MyA{Float32}, Val(2)) <: MyM{Float32}
    @test complex_arr_type(MyA{Float32}, Val(2)) <: MyM{ComplexF32}
    @test real_arr_type(MyA{ComplexF64,2}, Val(1)) <: MyV{Float64}
    @test complex_arr_type(MyA{ComplexF64,1}) <: MyV{ComplexF64}

    @test similar_arr_type(MyA{ComplexF64,1}, Int, Val(2)) <: MyM{Int}
    @test similar_arr_type(MyA{ComplexF64}, Float64, Val(2)) <: MyM{Float64}
    @test similar_arr_type(typeof(opt_cu(view(ones(10,10),2:5,2:5))), Float64, Val(1)) <: MyV{Float64}
    @test similar_arr_type(typeof(reinterpret(Int, opt_cu(ones(10)))), Float32, Val(2)) <: MyM{Float32}
    @test similar_arr_type(typeof(reshape(view(opt_cu(ones(25)),1:25), 5,5)), Int, Val(1)) <: MyV{Int}
end
