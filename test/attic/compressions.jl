
@testset "Compressions for sparse mps and mpo works" begin
    D = 16
    d = 2
    sites = collect(1:4)
    T = Float64

    Dcut = 8
    max_sweeps = 100
    tol = 1E-10

    ψ = rand(QMps{T}, sites, D, d)
    #W = rand(QMpo{T}, [1,2,3,4], 2, 4)

    bra = ψ
    ket = ψ
    #mpo = W

    @testset "Two mps representations are compressed to the same state" begin
        #χ = W * ψ
        #@test is_left_normalized(χ)

        #ϕ = copy(ψ)
        @test bond_dimension(bra) == max(D, d)
        @test bond_dimensions(bra) == [(1, d, D), (D, d, D), (D, d, D), (D, d, 1)]
        canonise!(bra, :left)
        #bra = QMps(ψ)

        #@time overlap, env = variational_compress!(bra, mpo, ket, tol, max_sweeps)
        #println(overlap)

        #ϕ = MPS(bra)
        #@time is_right_normalized(ϕ)
        #@test norm(χ) ≈ norm(bra) ≈ 1
    end
end
