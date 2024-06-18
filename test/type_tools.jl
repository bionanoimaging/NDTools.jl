@testset "Test Type Tools" begin
    sz = (11,12)
    @test real_arr_type(Array{Float32,2}) == Matrix{Float32}
    @test complex_arr_type(Array{Float32,1}, dims=2) == Matrix{ComplexF32}
    @test real_arr_type(Array{Float32}, dims=Val(2)) == Matrix{Float32}
    @test complex_arr_type(Array{Float32}, dims=Val(2)) == Matrix{ComplexF32}
    @test real_arr_type(Array{ComplexF64,2}, dims=Val(1)) == Vector{Float64}
    @test complex_arr_type(Array{ComplexF64,1}) == Vector{ComplexF64}
    @test similar_arr_type(Array{ComplexF64,1}, dims=Val(2), dtype=Int) == Matrix{Int}
    @test similar_arr_type(typeof(view(ones(10,10),2:5,2:5)), dims=Val(1)) == Vector{Float64}
    @test similar_arr_type(typeof(reinterpret(Int, ones(10))), dims=Val(2), dtype=Float32) == Matrix{Float32}
    @test similar_arr_type(typeof(reshape(view(ones(25),1:25), 5,5)), dims=Val(1), dtype=Int) == Vector{Int}
end
