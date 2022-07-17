using CUDA
using LinearAlgebra
using TensorOperations
using SpinGlassTensors

T = Float64

prmax = 4096
n, m, k = 32, 32, 4096

L = CUDA.rand(T, n, m, k);

pr = rand(1:prmax, k)

println("Naive:")
@time ipr0 = CUDA.CuArray(diagm(ones(maximum(pr)))[pr, :])

println("Less naive:")
@time ipr = CUDA.CuArray(Matrix(1.0I, prmax, prmax))[pr, :]

println("cuIdentity:")
@time ipr = cuIdentity(T, prmax)[pr, :]

println("cuProject")
@time id = cuProject(T, pr)

#println("Custom I + contraction:")
#@time add_project(L, pr, (16, 16))

#println("Contraction:")
#@time @tensor ret[x, y, r] := L[x, y, z] * ipr[z, r]

nothing
