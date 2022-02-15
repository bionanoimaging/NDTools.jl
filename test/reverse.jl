
@testset "reverse_view" begin

    a = randn((3,4,3,5))    
    f(a, dims) = @test reverse(a, dims=dims) == reverse_view(a, dims=dims)

    @test reverse(a) == reverse_view(a)
    for i in [1,2,4, (1,2), (3,4), (1,2,3,4), (1,4)]
        f(a, i)
    end
end
