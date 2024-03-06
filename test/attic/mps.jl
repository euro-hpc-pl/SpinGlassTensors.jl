@testset "QMps" begin

    T = Float64
    D = 16

    sites = [1, 3 // 2, 2, 5 // 2, 3, 7 // 2, 4]
    d = [1, 2, 2, 2, 4, 2, 2]

    id = Dict(j => d[i] for (i, j) in enumerate(sites))

    @testset "Random QMps with varying physical dimension" begin
        ψ = rand(QMps{T}, id, D)

        @testset "has correct number of sites" begin
            @test length(ψ) == maximum(sites)
            @test size(ψ) == (maximum(sites),)
        end

        @testset "has correct type" begin
            @test eltype(ψ) == T
        end

        @testset "has correct rank" begin
            @test rank(ψ) == Tuple(d)
        end

        @testset "has correct bonds" begin
            @test bond_dimension(ψ) ≈ D
            @test bond_dimensions(ψ) == [
                (1, d[1], D),
                (D, d[2], D),
                (D, d[3], D),
                (D, d[4], D),
                (D, d[5], D),
                (D, d[6], D),
                (D, d[7], 1),
            ]
            @test verify_bonds(ψ) === nothing
        end

        @testset "is equal to itself" begin
            @test ψ == ψ
        end
    end
end
