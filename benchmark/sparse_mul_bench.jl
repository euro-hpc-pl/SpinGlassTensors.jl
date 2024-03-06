using CUDA
using LinearAlgebra
using SparseArrays

function dense_x_CSC(Md::DenseCuMatrix{T}, Mcsc::CUSPARSE.CuSparseMatrixCSC{T}) where {T}
    ret = CUDA.zeros(T, size(Md, 1), size(Mcsc, 2))
    CUSPARSE.mm!('N', 'N', one(T), Mcsc, Md, zero(T), ret, 'O')
    ret
end

T = Float64
nnz = 2^14
Val = CUDA.rand(T, nnz)
Ptr = CuArray(1:nnz+1)
Ind = CuArray(rand(1:nnz, nnz))

Mcsr = CUSPARSE.CuSparseMatrixCSR(Ptr, Ind, Val, (nnz, nnz))
Md = CUDA.rand(T, nnz, nnz)
Mcsc = CUSPARSE.CuSparseMatrixCSC(Ptr, Ind, Val, (nnz, nnz))

@time CUDA.@sync X = Mcsr * Md
#@time CUDA.@sync Y = dense_x_CSC(Md, Mcsc)
@time CUDA.@sync Z = (Mcsc' * Md')'

println()

@time CUDA.@sync X = Mcsr * Md
#@time CUDA.@sync Y = dense_x_CSC(Md, Mcsc)
@time CUDA.@sync Z = (Mcsc' * Md')'

nothing
