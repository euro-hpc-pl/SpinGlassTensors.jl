using CUDA
using LinearAlgebra
using TensorOperations
using SpinGlassTensors

T = Float64

prmax = 4096
n, m, k = 32, 32, 4096

L = CUDA.rand(T, n, m, k);

pr = rand(1:prmax, k)

@time ipr = CUDA.CuArray(diagm(ones(maximum(pr)))[pr, :])
@time @tensor ret[x, y, r] := L[x, y, z] * ipr[z, r]

println("Custom:")

@time add_project(L, pr, (16, 16))

nothing
