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


@testset "Results of right_env are correctly cached" begin
    empty!(memoize_cache(right_env))

    Random.seed!(42)

    ψ = randn(MPS{T}, sites, D, d) 
    ϕ = copy(ψ)

    W = randn(MPO{T}, sites, D, d)
    V = copy(W)

    σ = [2, 1, 1, 2, 1]
    η = copy(σ)
    
    envs = [
        right_env(mps, mpo, v) for mps in (ψ, ϕ) for mpo ∈ (W, V) for v ∈ (σ, η)
    ]
    
    @testset "Calls made with equal arguments results in a cache hit" begin
        # Should result in 6 calls total (top-level + 5 recursive)
        @test length(memoize_cache(right_env)) == 6
    end

    @testset "Cached results are equal to the ones computed during first call" begin
        @test all(env->env==envs[1], envs)
    end
end
