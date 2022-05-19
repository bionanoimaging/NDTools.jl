@testset "Base extensions" begin
    

    @testset "sumdropdims" begin 
        arr = randn((2,2,2,2,2))

        function test_f(f, arr, dims)
            @test dropdims(sum(f, arr, dims=dims), dims=dims) == sumdropdims(f, arr, dims=dims)
            @test dropdims(sum(arr, dims=dims), dims=dims) == sumdropdims(arr, dims=dims)
        end

        test_f(x -> x^2, arr, (1,2))
        test_f(x -> x^2, arr, (1,3))
        test_f(x -> x^2, arr, (2,4))
        test_f(x -> x^2, arr, (1,2,3,4, 5))
        test_f(x -> x^2, arr, (1,2,3))
        
        @test ndims(sumdropdims(arr, dims=(1,2,3,4,5))) == 0
    end
end
