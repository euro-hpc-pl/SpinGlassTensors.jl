
@testset "QMps" begin

D = 16
d = 4
sites = [1,2,3,4]

T = Float64

@testset "Random QMps" begin

    ψ = rand(QMps{T}, sites, D, d)

    @testset "has correct number of sites" begin
        @test length(ψ) == length(sites) 
        @test size(ψ) == (length(sites), )      
    end
 
    @testset "has correct type" begin
        @test eltype(ψ) == T       
    end

    @testset "has correct rank" begin
        @test rank(ψ) == Tuple(fill(d, length(sites)))      
    end

    @testset "has correct bonds" begin
        @test bond_dimension(ψ) ≈ D 
        @test bond_dimensions(ψ) == [(1, d, D), (D, d, D), (D, d, D), (D, d, 1)]    
        @test verify_bonds(ψ) === nothing
    end

    @testset "is equal to itself" begin
        @test ψ == ψ
    end

end 

# @testset "Random MPO with the same physical dimension" begin
#     sites_aux = [1//3, 1//2, 2//3]
#     d_aux = 4

#     W = random_QMpo(sites, D, d)
#     Z = random_QMpo(sites, D, d, sites_aux, d_aux)
#     @testset "has correct number of sites" begin
#         @test length(W) == length(sites)
#         @test length(Z) == length(sites)
#         @test size(W) == (length(sites), )   
#         @test size(Z) == (length(sites), )      
#     end
 
    # @testset "has correct type" begin
    #     @test eltype(W) == T       
    # end

    # @testset "is equal to itself" begin
    #     @test W == W
    #     @test W ≈ W
    #     @test Z == Z
    #     @test Z ≈ Z
    # end

    # @testset "is equal to its copy" begin
    #     U = copy(W)
    #     @test U == W
    #     @test U ≈ W
    # end
# end 

end
