T = Float64
D = 16

sites = [1, 3//2, 2, 5//2, 3, 7//2, 4]
d = [1, 2, 2, 2, 4, 2, 2]

id = Dict(j => d[i] for (i, j) in enumerate(sites))

@testset "Random QMps" begin
    ψ = rand(QMps{T}, id, D)
    ϕ = rand(QMps{T}, id, D)

    ψ = move_to_CUDA!(ψ)
    ϕ = move_to_CUDA!(ϕ)

    @testset "is left normalized" begin
        canonise!(ψ, :left)
        #@test is_left_normalized(ψ)
    end

    @testset "is right normalized" begin
        canonise!(ϕ, :right)
        #@test is_right_normalized(ϕ)
    end
end
