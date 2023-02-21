l = 4
D1 = [1, 2, 2, 1]
D2 = [2, 2, 2, 16]
D3 = [2, 2, 2, 16]
D4 = [2, 2, 2, 1]
S = Float64
map1 = MpoTensor(TensorMap{S}(Dict(-1//2 => rand(CentralTensor{S}, [1, 1, 1, 1, 1, 1, 1, 1]), 0 => rand(SiteTensor{S}, PoolOfProjectors{Integer}(), l, D1))))
map2 = MpoTensor(TensorMap{S}(Dict(-1//2 => rand(CentralTensor{S}, [1, 1, 1, 1, 1, 1, 1, 1]), 0 => rand(SiteTensor{S}, PoolOfProjectors{Integer}(), l, D2))))
map3 = MpoTensor(TensorMap{S}(Dict(-1//2 => rand(CentralTensor{S}, [1, 1, 1, 1, 1, 1, 1, 1]), 0 => rand(SiteTensor{S}, PoolOfProjectors{Integer}(), l, D3))))
map4 = MpoTensor(TensorMap{S}(Dict(-1//2 => rand(CentralTensor{S}, [1, 1, 1, 1, 1, 1, 1, 1]), 0 => rand(SiteTensor{S}, PoolOfProjectors{Integer}(), l, D4))))
# mpomap = MpoTensorMap{S}(Dict(1 => map1, 2 => map2, 3 => map3, 4 => map4))
mpomap = Dict(1 => map1, 2 => map2, 3 => map3, 4 => map4)

D = 4
sites = [1, 2, 3, 4]
d = [1, 2, 2, 2]
id = Dict(j => d[i] for (i, j) in enumerate(sites))

@testset "Random QMpo with varying physical dimension" begin
    W = rand(QMpo{S}, mpomap)

    @test length(W) == 4
    @test bond_dimension(W) == 2
end

@testset "Compressions for sparse mps and mpo works" begin
    W = rand(QMpo{S}, mpomap)
    println(bond_dimensions(W))
    ψ = rand(QMps{S}, id, D)
    println(bond_dimensions(ψ))
    ϕ = rand(QMps{S}, id, D)

    Dcut = 8
    max_sweeps = 100
    tol = 1E-10

    # χ = W * ψ
    # @time overlap, env = variational_compress!(bra, mpo, ket, tol, max_sweeps)
end
