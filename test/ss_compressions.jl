@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 100
    T = Float64
    
    Dcut = 64
    max_sweeps = 10
    tol = 1E-4

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    ket = QMps(ψ)
    mpo = QMpo(W)

    @testset "Two mps representations are compressed to the same state" begin 
        χ = W * ψ
        @time overlap = variational_compress!(χ, Dcut, tol, max_sweeps)
        @test is_left_normalized(χ)
        println(overlap)

        ϕ = copy(ψ)
        canonise!(ϕ, :left)
        bra = QMps(ϕ)

        @time overlap = compress_twosite!(bra, mpo, ket, Dcut, tol, max_sweeps)
        println(overlap)
        
        ϕ = MPS(bra)
        @time is_right_normalized(ϕ)
        @test norm(χ) ≈ norm(ϕ) ≈ 1
    end
end
