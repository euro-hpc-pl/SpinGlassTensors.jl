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
        1 =>  [0.601308  0.067023;;;  0.807881  0.478226], 
        3//2 => [0.198992   -0.117862; 0.0759264  -0.165057;;; -0.473871   0.690909;
        -0.218285  -0.111557;;;  -0.139789  0.0681208;
        -0.221155  0.533701;;; -0.0341383   0.11568;
        0.106109   -0.139025], 
        2 => [0.253553;  0.106135;;;
        0.106135;  0.149019;;;]
    )
    ψ = QMps(svd_mps) 
    canonise!(ψ, :left)
    A = measure_spectrum(ψ)
    @test A[1] == [1.0] 
    @test A[3//2] == [0.9809920291334959, 0.19404803213778352] #[0.908877407277003 0.6523067447788067]
    @test A[2] == [0.9646070321115633, 0.2636916259590388] #[0.31959234331055053 0.08297946199028772]
end