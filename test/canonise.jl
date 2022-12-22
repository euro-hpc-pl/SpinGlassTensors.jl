@testset "Canonise" begin

    D = 16
    d = 4
    sites = [1,2,3,4]
    
    T = Float64
    

@testset "Random QMps" begin

    ψ = rand(QMps{T}, sites, D, d)
    ϕ = rand(QMps{T}, sites, D, d)

    @testset "is left normalized" begin
        canonise!(ψ, :left)
        @test is_left_normalized(ψ)
    end

    @testset "is right normalized" begin
        canonise!(ϕ, :right)
        @test is_right_normalized(ϕ)
    end

end
end