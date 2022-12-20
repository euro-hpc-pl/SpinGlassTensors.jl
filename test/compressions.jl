@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = collect(1:100)
    T = Float64

    Dcut = 8
    max_sweeps = 100
    tol = 1E-10

    ψ = random_QMps(sites, D, d)
    W = random_QMpo(sites, D, d, [1//2], 2)

    ket = ψ
    mpo = W

    @testset "Two mps representations are compressed to the same state" begin
        χ = W * ψ
        @test is_left_normalized(χ)

        ϕ = copy(ψ)
        canonise!(ϕ, :left)
        bra = QMps(ϕ)

        @time overlap, env = variational_compress!(bra, mpo, ket, tol, max_sweeps)
        println(overlap)

        ϕ = MPS(bra)
        @time is_right_normalized(ϕ)
        @test norm(χ) ≈ norm(ϕ) ≈ 1
    end
end
