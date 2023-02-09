using CUDA
using LinearAlgebra
using SparseArrays

function dense_x_sparse(Md::DenseCuMatrix{T}, Mcsr::CUSPARSE.CuSparseMatrixCSR{T}) where T
    ret = CUDA.zeros(T, size(Md, 1), size(Mcsr, 2))
    CUSPARSE.mm!('N', 'N', one(T), Mcsr, Md, zero(T), ret, 'O')
    ret
end

function dense_x_sparse(Md::DenseCuMatrix{T}, Mcsc::CUSPARSE.CuSparseMatrixCSC{T}) where T
    ret = CUDA.zeros(T, size(Mcsc, 1), size(Md, 2))
    CUSPARSE.mm!('N', 'N', one(T), Mcsc, Md, zero(T), ret, 'O')
    ret
end

CUDA.allowscalar(true)

T = Float64
nnz = 3
Val = CUDA.rand(T, nnz)
Ptr = CuArray(1:nnz+1)
Ind = CuArray(rand(1:nnz, nnz))

Mcsr = CUSPARSE.CuSparseMatrixCSR(sprand(nnz, nnz, 0.8))#Ptr, Ind, Val, (nnz, nnz))
Md = CUDA.rand(T, nnz, nnz)
Mcsc = CUSPARSE.CuSparseMatrixCSC(sprand(nnz, nnz, 0.8))#Ptr, Ind, Val, (nnz, nnz))

X = Mcsc * Md # OK
Y = Md * Mcsr # NO - this returns a CPU matrix.

@assert X ≈ dense_x_sparse(Md, Mcsc) # OK
@assert Y ≈ dense_x_sparse(Md, Mcsr) |> Array # NO

#println(Y)
#println(dense_x_sparse(Md, Mcsr) |> Array)
