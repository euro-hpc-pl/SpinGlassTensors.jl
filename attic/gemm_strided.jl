using CUDA
using NNlib
using NNlibCUDA

m, n, k = 256, 256, 4096

A = CUDA.rand(m, n, k)
B = CUDA.rand(m, n, k)

@time D = A ⊠ B
@time C = CUDA.CUBLAS.gemm_strided_batched('N', 'N', A, B)

@assert Array(C) ≈ Array(D)
