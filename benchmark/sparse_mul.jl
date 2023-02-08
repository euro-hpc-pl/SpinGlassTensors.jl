using CUDA
using LinearAlgebra

function dense_x_sparse(Md::DenseCuMatrix{T}, Mcsr_csc) where T
    ret = CUDA.zeros(T, size(Md, 1), size(Mcsr_csc, 2))
    CUSPARSE.mm!('N', 'N', one(T), Mcsr_csc, Md, zero(T), ret, 'O')
    ret
end

CUDA.allowscalar(true)

T = Float64
nnz = 1
Val = CUDA.rand(T, nnz)
Ptr = CuArray(1:nnz+1)
Ind = CuArray(rand(1:nnz, nnz))

Mcsr = CUSPARSE.CuSparseMatrixCSR(Ptr, Ind, Val, (nnz, nnz))
Md = CUDA.rand(T, nnz, nnz)
Mcsc = CUSPARSE.CuSparseMatrixCSC(Ptr, Ind, Val, (nnz, nnz))

X = Mcsc * Md # OK
Y = Md * Mcsr # NO - this returns a CPU matrix.

@assert X ≈ dense_x_sparse(Md, Mcsc)
#@assert Y ≈ dense_x_sparse(Md, Mcsr) |> Array

println(Y)
println(dense_x_sparse(Md, Mcsr) |> Array)
