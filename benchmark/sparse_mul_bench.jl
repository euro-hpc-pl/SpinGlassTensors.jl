using CUDA
using LinearAlgebra

function dense_x_CSR(Md::DenseCuMatrix{T}, Mcsr::CUSPARSE.CuSparseMatrixCSR{T}) where T
    ret = CUDA.zeros(T, size(Md, 1), size(Mcsr, 2))
    CUSPARSE.mm!('T', 'T', one(T), Mcsr, Md, zero(T), ret, 'O')
    ret'
end


CUDA.allowscalar(true)

T = Float64
nnz = 2^14
Val = CUDA.rand(T, nnz)
Ptr = CuArray(1:nnz+1)
Ind = CuArray(rand(1:nnz, nnz))

Mcsr = CUSPARSE.CuSparseMatrixCSR(Ptr, Ind, Val, (nnz, nnz))
Md = CUDA.rand(T, nnz, nnz)
Mcsc = CUSPARSE.CuSparseMatrixCSC(Ptr, Ind, Val, (nnz, nnz))

@time X = Mcsc * Md
@time Y = dense_x_CSR(Md, Mcsr)

println()

@time X = Mcsc * Md
@time Y = dense_x_CSR(Md, Mcsr)

nothing
