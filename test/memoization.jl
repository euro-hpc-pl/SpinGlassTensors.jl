using Memoize
using Random
using Test
using SpinGlassTensors


D = 10
d = 3
sites = 5
T = Float64

@testset "Calling left_env with equal arguments results in a cache hit" begin
    # Cache is global, therefore we need to reset it to obtain meaningful
    # results here.
    empty!(memoize_cache(left_env))

    Random.seed!(69)

    ψ = randn(MPS{T}, sites, D, d) 
    ϕ = copy(ψ)

    σ = [1, 2, 3]
    η = [1, 2, 3]

    # Should result in 4 calls total (top-level + 3 recursive)
    env_1 = left_env(ψ, σ) 
    @test length(memoize_cache(left_env)) == 4

    env_2 = left_env(ψ, η)

    env_3 = left_env(ϕ, σ)
    
    env_4 = left_env(ϕ, η)
    
    @test env_1 == env_2 == env_3 == env_4

    # No additional calls should be made, thus size of cache should still be equal to 4
    @test length(memoize_cache(left_env)) == 4
end