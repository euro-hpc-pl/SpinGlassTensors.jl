
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
    
    @testset "dot product of MPS with MPO" begin
        B = randn(Float64, 4,2,3)
        A = randn(Float64, 2,2)  
        C = randn(Float64, 2,2,2,2)

        @testset "contract_left gives correct sizes" begin 
            @test size(contract_left(B,A)) == (4,2,3)
        end

        @testset "contract_up gives correct sizes" begin 
            @test size(contract_up(B,A)) == (4,2,3)
            @test size(contract_up(B,C)) == (8,2,6)
        end

        @testset "dot product of AbstractMpo and Mps" begin
            O2 = randn(MPO{T}, sites, D, d)
            D = dot(O2, ψ)
            @test size(D[1]) == (1, 2, 4)
            @test size(D[2]) == (4, 2, 1)
        end

    end

end