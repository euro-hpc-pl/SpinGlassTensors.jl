@testset "MPS" begin

    D = 10
    d = 4
    sites = 5
    T = ComplexF64

    @testset "Random MPS with the same physical dimension" begin

        ψ = randn(MPS{T}, sites, D, d)

        @testset "has correct number of sites" begin
            @test length(ψ) == sites
            @test size(ψ) == (sites,)
        end

        @testset "has correct type" begin
            @test eltype(ψ) == T
        end

        @testset "has correct rank" begin
            @test rank(ψ) == Tuple(fill(d, sites))
        end

        @testset "has correct bonds" begin
            @test bond_dimension(ψ) ≈ D
            @test verify_bonds(ψ) === nothing
        end

        @testset "is equal to itself" begin
            @test ψ == ψ
            @test ψ ≈ ψ
        end

        @testset "is equal to its copy" begin
            ϕ = copy(ψ)
            @test ϕ == ψ
            @test ϕ ≈ ψ
        end
    end

    @testset "Random MPS with varying physical dimension" begin

        dims = (3, 2, 5, 4)
        ψ = randn(MPS{T}, D, dims)

        @testset "has correct number of sites" begin
            n = length(dims)
            @test length(ψ) == n
            @test size(ψ) == (n,)
        end

        @testset "has correct type" begin
            @test eltype(ψ) == T
        end

        @testset "has correct rank" begin
            @test rank(ψ) == dims
        end

        @testset "has correct bonds" begin
            @test bond_dimension(ψ) ≈ D
            @test verify_bonds(ψ) === nothing
        end

        @testset "is equal to itself" begin
            @test ψ == ψ
            @test ψ ≈ ψ
        end

        @testset "is equal to its copy" begin
            ϕ = copy(ψ)
            @test ϕ == ψ
            @test ϕ ≈ ψ
        end
    end

    @testset "Random MPO with the same physical dimension" begin

        W = randn(MPO{T}, sites, D, d)

        @testset "has correct number of sites" begin
            @test length(W) == sites
            @test size(W) == (sites,)
        end

        @testset "has correct type" begin
            @test eltype(W) == T
        end

        @testset "is equal to itself" begin
            @test W == W
            @test W ≈ W
        end

        @testset "is equal to its copy" begin
            U = copy(W)
            @test U == W
            @test U ≈ W
        end
    end

    @testset "Random MPO with varying physical dimension" begin

        dims = (3, 2, 5, 4)
        W = randn(MPO{T}, D, dims)

        @testset "has correct number of sites" begin
            n = length(dims)
            @test length(W) == n
            @test size(W) == (n,)
        end

        @testset "has correct type" begin
            @test eltype(W) == T
        end

        @testset "is equal to itself" begin
            @test W == W
            @test W ≈ W
        end

        @testset "is equal to its copy" begin
            U = copy(W)
            @test U == W
            @test U ≈ W
        end
    end

    @testset "MPS from tensor" begin
        ϵ = 1E-14

        dims = (2, 3, 4, 3, 5)
        sites = length(dims)
        A = randn(T, dims)

        ψ = MPS(A, :right)

        @test norm(ψ) ≈ 1
        @test_nowarn verify_bonds(ψ)
        @test_nowarn verify_physical_dims(ψ, dims)
        @test is_right_normalized(ψ)

        B = randn(T, dims...)
        ϕ = MPS(B, :left)

        @test norm(ϕ) ≈ 1
        @test_nowarn verify_bonds(ϕ)
        @test_nowarn verify_physical_dims(ϕ, dims)
        @test is_left_normalized(ϕ)

        χ = MPS(A, :left)

        @test norm(χ) ≈ 1
        @test abs(1 - abs(dot(ψ, χ))) < ϵ
    end

end


@testset "Objects with equal tensors have the same hash" begin
    D = 10
    d = 4
    sites = 5
    T = ComplexF64

    ψ = randn(MPS{T}, sites, D, d)
    ϕ = copy(ψ)

    W = randn(MPO{T}, sites, D, d)
    V = copy(W)

    @testset "Equal MPSs have the same hash" begin
        @test hash(ψ) == hash(ϕ)
    end

    @testset "Equal MPOs have the same hash" begin
        @test hash(W) == hash(V)
    end

    @testset "Equal tuples with MPS and MPO have the same hash" begin
        tuple_1 = (ψ, W, [1, 2, 3])
        tuple_2 = (ϕ, V, [1, 2, 3])
        @test tuple_1 == tuple_2
        @test hash(tuple_1) == hash(tuple_2)
    end
end
