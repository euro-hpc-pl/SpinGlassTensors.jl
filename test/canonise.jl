T = Float64
D = 16

sites = [1, 3//2, 2, 5//2, 3, 7//2, 4]
d = [1, 2, 2, 2, 4, 2, 2]

id = Dict(j => d[i] for (i, j) in enumerate(sites))

@testset "Random QMps" begin
    for toCUDA ∈ (true, false)
        ψ = rand(QMps{T}, id, D)
        ϕ = rand(QMps{T}, id, D)
        @test is_consistent(ψ)
        @test is_consistent(ϕ)

        if toCUDA
            ψ = move_to_CUDA!(ψ)
            ϕ = move_to_CUDA!(ϕ)
            @test is_consistent(ψ)
            @test is_consistent(ϕ)
        end

        @testset "is left normalized" begin
            canonise!(ψ, :left)
            @test is_consistent(ψ)
            @test is_left_normalized(ψ)
            @test dot(ψ, ψ) ≈ one(T)
        end

        @testset "is right normalized" begin
            canonise!(ϕ, :right)
            @test is_consistent(ϕ)
            @test is_right_normalized(ϕ)
            @test dot(ϕ, ϕ) ≈ one(T)
        end
    end
end

@testset "Measure spectrum" begin
    svd_mps = TensorMap{T}(
        1 => [1.0;;;0.0], 3//2 => [1.0 0.0;;; 1.0 0.0], 
        2 => permutedims([1.0 0.0; 1.0 0.0 ;;;], (1,3,2))
    )
    ψ = QMps(svd_mps) 
    canonise!(ψ, :left)
    A = measure_spectrum(ψ)
    @test [A[i]==[1.0] for i in keys(A)] == Bool[1, 1, 1]
end