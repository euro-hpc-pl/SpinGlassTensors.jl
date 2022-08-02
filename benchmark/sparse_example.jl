
using CUDA
using CUDA.CUSPARSE
using SparseArrays

T = Float64
n = 1000

a = rand(T, n, n); a[a .< 0.9] .= T(0)
a = CuSparseMatrixCSC(sparse(a))

b = CUDA.rand(T, n, n);

@time a * b
nothing
