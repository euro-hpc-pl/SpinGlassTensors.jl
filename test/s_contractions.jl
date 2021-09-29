@testset "Contraction" begin
    D = 2
    d = 2
    sites = 2
    T = Float64

    ψ = Mps(randn(MPS{T}, sites, D, d))
    ϕ = Mps(randn(MPS{T}, sites, D, d))
    O1 = Mpo(randn(MPO{T}, sites, D, d))

    @testset "dot products of MPS" begin
        @testset "is equal to itself" begin
            @test dot(ψ, ψ) ≈ dot(ψ, ψ)
        end

        @testset "change of arguments results in conjugation" begin
            @test dot(ψ, ϕ) ≈ conj(dot(ϕ, ψ))
        end

        @testset "norm is 2-norm" begin
            @test norm(ψ) ≈ sqrt(abs(dot(ψ, ψ)))
        end

        @testset "renormalizations" begin
            ψ.tensors[ψ.sites[end]] *= 1/norm(ψ)
            @test dot(ψ, ψ) ≈ 1

            ϕ.tensors[ψ.sites[1]] *= 1/norm(ϕ)
            @test dot(ϕ, ϕ) ≈ 1
        end
    end 

end