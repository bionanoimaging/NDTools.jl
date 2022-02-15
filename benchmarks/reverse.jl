using BenchmarkTools, NDTools


function compare_reverse(arr, dims=Tuple(1:ndims(arr)))
    print("## runtime with collect\nsize(arr) = $(size(arr))\n")
    print("\treverse:\t")
    @btime reverse($arr, dims=$dims)
    print("\treverse_view:\t")
    @btime collect(reverse_view($arr, dims=$dims))
    print("\n")
end

function compare_reverse_broadcast(arr, dims=Tuple(1:ndims(arr)))
    print("## Broadcast \nsize(arr) = $(size(arr))\n")
    print("\treverse:\t")
    arr2 = copy(arr)
    @btime $arr2 .= abs2.(reverse!($arr, dims=$dims))
    print("\treverse_view:\t")
    @btime $arr2 .= abs2.(reverse_view($arr, dims=$dims))
    print("\n")
end 

function main()
    compare_reverse_broadcast(randn(10, 10))
    compare_reverse_broadcast(randn(1013, 1024))
    compare_reverse_broadcast(randn(512, 13))
    compare_reverse_broadcast(randn(512, 133, 231,2))
    compare_reverse_broadcast(randn(512, 133, 231,2), (1,4))

    compare_reverse(randn(10, 10))
    compare_reverse(randn(1013, 1024))
    compare_reverse(randn(512, 13))
    compare_reverse(randn(512, 133, 231,2))
    compare_reverse(randn(512, 133, 231,2), (1,4))
end
