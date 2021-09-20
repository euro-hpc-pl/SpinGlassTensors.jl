@testset "Canonisation and Compression" begin

D = 16
Dcut = 8

d = 2
sites = 100

T = Float64

ψ = randn(MPS{T}, sites, D, d)
ϕ = randn(MPS{T}, sites, D, d)
χ = randn(MPS{T}, sites, D, d)
Φ = randn(MPS{T}, sites, D, d)



@testset "Canonisation (left)" begin
    b = canonise!(ψ, :left)
    @test is_left_normalized(ψ)
    @test dot(ψ, ψ) ≈ 1
end


@testset "Canonisation (right)" begin
    b = canonise!(ϕ, :right)
    @test is_right_normalized(ϕ)
    @test dot(ϕ, ϕ) ≈ 1
end

@testset "Copy and canonise twice" begin
    ψ̃ = copy(ψ)
    @test ψ̃ == ψ
    for (direction, predicate) ∈ ((:left, is_left_normalized), (:right, is_right_normalized))
        canonise!(ψ, direction, Dcut)
        canonise!(ψ̃, direction, Dcut)

        @test predicate(ψ)
        @test predicate(ψ̃)
        @test bond_dimension(ψ̃) == bond_dimension(ψ) 
        @test all(size(A) == size(B) for (A, B) ∈ zip(ψ, ψ̃))
        @test typeof(ψ̃) == typeof(ψ)
        @test norm(ψ) ≈ norm(ψ̃) ≈ 1 
        @test abs(ψ * ψ̃) ≈ abs(ψ̃ * ψ) ≈ 1 
    end    
end

@testset "Cauchy-Schwarz inequality (after truncation)" begin
    @test abs(dot(ϕ, ψ)) <= norm(ϕ) * norm(ψ)
end

@testset "Truncation (SVD, right)" begin
    canonise!(ψ, :right, Dcut)
    @test is_right_normalized(ψ)
    @test norm(ψ) ≈ 1
end

@testset "Truncation (SVD, left)" begin
    canonise!(ψ, :left, Dcut)
    @test is_left_normalized(ψ)
    @test norm(ψ) ≈ 1
end


@testset "<left|right>" begin
    ϵ = 1E-10
    ψ = randn(MPS{T}, sites, D, d)

    l = copy(ψ)
    r = copy(ψ)

    canonise!(l, :left)
    @test is_left_normalized(l)

    canonise!(r, :right)
    @test is_right_normalized(r)

    @test dot(l, l) ≈ 1
    @test dot(r, r) ≈ 1

    @test abs(1 - abs(dot(l, r))) < ϵ
end


@testset "Variational compression" begin
    Dcut = 8
    tol = 1E-10
    max_sweeps = 100

    Ψ = copy(Φ)
    canonise!(Ψ, :left)

    compress!(Φ, Dcut, tol, max_sweeps)

    println(dot(Φ, Ψ))

    @test norm(Φ) ≈ 1
    @test is_left_normalized(Φ)
    @test is_right_normalized(Φ) == false
end

end
