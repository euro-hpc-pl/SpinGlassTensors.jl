@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = 100
    T = Float64
    
    Dcut = 8
    max_sweeps = 100
    tol = 1E-10

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    ket = QMPS(ψ)
    mpo = QMPO(W)

    @testset "Two mps representations are compressed to the same state" begin 
        χ = W * ψ
        overlap = compress!(χ, Dcut, tol, max_sweeps)
        @test is_left_normalized(χ)

        ϕ = copy(ψ)
        canonise!(ϕ, :left)
        bra = QMPS(ϕ)

        overlap = compress!(bra, mpo, ket, Dcut, tol, max_sweeps)
        
        ϕ = MPS(bra)
        is_right_normalized(ϕ)
        @test norm(χ) ≈ norm(ϕ) ≈ 1
    end
end
