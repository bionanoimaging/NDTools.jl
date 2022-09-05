@testset "Test Type Tools" begin
    sz = (11,12)
    @test real_arr_type(Array{Float32,2}) == Matrix{Float32}
    @test complex_arr_type(Array{Float32,1}, dims=2) == Matrix{ComplexF32}
    @test real_arr_type(Array{Float32}, dims=2) == Matrix{Float32}
    @test complex_arr_type(Array{Float32}, dims=2) == Matrix{ComplexF32}
    @test real_arr_type(Array{ComplexF64,2}, dims=1) == Vector{Float64}
    @test complex_arr_type(Array{ComplexF64,1}) == Vector{ComplexF64}
    @test similar_arr_type(Array{ComplexF64,1}, dims=2, dtype=Int) == Matrix{Int}
    @test similar_arr_type(Array{ComplexF64}, dims=2, dtype=Int) == Matrix{Int}
end
