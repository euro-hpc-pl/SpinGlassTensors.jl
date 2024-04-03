@testset "Environment" begin
    sites = [1, 1 // 2, 2, 3, 7 // 2, 4, 5, 6]
    site = 3
    @testset "left_nbrs_site gives correct left neighbor of a given site" begin
        @test left_nbrs_site(site, sites) == 2
    end
    @testset "left_nbrs_site gives correct right neighbor of a given site" begin
        @test right_nbrs_site(site, sites) == 7 // 2
    end
end
