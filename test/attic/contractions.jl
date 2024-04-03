@testset "Contraction" begin
    D = 2
    d = 2
    sites = collect(1:2)
    T = Float64

    ψ = random_QMps(sites, D, d)
    ϕ = random_QMps(sites, D, d)
    O1 = random_QMpo(sites, D, d)

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
            ψ.tensors[ψ.sites[end]] *= 1 / norm(ψ)
            @test dot(ψ, ψ) ≈ 1

            ϕ.tensors[ψ.sites[1]] *= 1 / norm(ϕ)
            @test dot(ϕ, ϕ) ≈ 1
        end
    end

    @testset "dot product of MPS with MPO" begin
        B = randn(Float64, 4, 2, 3)
        A = randn(Float64, 2, 2)
        C = randn(Float64, 2, 2, 2, 2)
        O2 = random_QMpo(sites, D, d)

        @testset "contract_left gives correct sizes" begin
            @test size(contract_left(B, A)) == (4, 2, 3)
        end

        @testset "contract_tensors43 gives correct sizes" begin
            @test size(contract_tensors43(C, B)) == (8, 2, 6)
        end

        # @testset "dot product of QMpo and QMps" begin
        #     D = dot(O2, ψ)
        #     E = dot(O1, ψ)
        #     @test size(D[1]) == size(E[1]) == (1, 2, 4)
        #     @test size(D[2]) == size(E[2]) == (4, 2, 1)
        # end

        @testset "dot product of QMps and QMpo" begin
            F = dot(ψ, O2)
            G = dot(ψ, O1)
            @test size(F[1]) == size(G[1]) == (1, 2, 4)
            @test size(F[2]) == size(G[2]) == (4, 1, 2)
        end
    end
end
