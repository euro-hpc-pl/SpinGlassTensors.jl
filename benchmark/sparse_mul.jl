using CUDA
using LinearAlgebra

function CUDA.:*(Md::DenseCuMatrix{T}, Mcsr::CUSPARSE.CuSparseMatrixCSR{T}) where T
    ret = CUDA.zeros(T, size(Md, 1), size(Mcsr, 2))
    CUSPARSE.mm!('T', 'T', one(T), Mcsr, Md, zero(T), ret, 'O')
    ret'
end

function CUDA.:*(Md::DenseCuMatrix{T}, Mcsc::CUSPARSE.CuSparseMatrixCSC{T}) where T
    ret = CUDA.zeros(T, size(Mcsc, 1), size(Md, 2))
    CUSPARSE.mm!('N', 'N', one(T), Mcsc, Md, zero(T), ret, 'O')
    ret
end

CUDA.allowscalar(true)

T = Float64
nnz = 100
Val = CUDA.rand(T, nnz)
Ptr = CuArray(1:nnz+1)
Ind = CuArray(rand(1:nnz, nnz))

Mcsr = CUSPARSE.CuSparseMatrixCSR(Ptr, Ind, Val, (nnz, nnz))
Md = CUDA.rand(T, nnz, nnz)
Mcsc = CUSPARSE.CuSparseMatrixCSC(Ptr, Ind, Val, (nnz, nnz))

X = Mcsc * Md
Y = Md * Mcsr

@assert X ≈ CuArray(Mcsc) * Md
@assert Y ≈ Md * CuArray(Mcsr)
