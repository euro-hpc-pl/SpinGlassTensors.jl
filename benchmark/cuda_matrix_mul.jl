using CUDA
using LinearAlgebra

CUDA.allowscalar(false)

nnz = 100
Val = CUDA.rand(Float64, nnz)
Ptr = CuArray(1:nnz+1)
Ind = CuArray(rand(1:100, nnz))

A = CUDA.CUSPARSE.CuSparseMatrixCSR(Ptr, Ind, Val, (100, 100))
B = CUDA.rand(Float64, 100, 100)
C = CUDA.CUSPARSE.CuSparseMatrixCSC(Ptr, Ind, Val, (100, 100))

A*B # no scalar indexing
CUDA.@allowscalar B*A # scalar indexing

C*B # no scalar indexing
CUDA.@allowscalar B*C # scalar indexing

A'*B # no scalar indexing
CUDA.@allowscalar B*A' # scalar indexing

transpose(A)*B # no scalar indexing
CUDA.@allowscalar B*transpose(A) # scalar indexing
# problem is when we multiply dense x sparse

 D = rand(Float64, (100, 100))
 CUDA.@allowscalar D*A # scalar indexing
