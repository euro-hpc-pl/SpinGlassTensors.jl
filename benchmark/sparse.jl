using CUDA
using LinearAlgebra
using TensorOperations
using TensorCast
using SpinGlassTensors
using SparseArrays

CUDA.allowscalar(false)

T = Float64

prmax = 512
n, m, k = 4, 4, prmax ^ 2

Lr = rand(T, n, m, k)
pr = rand(1:prmax, k)

Lr_d = CUDA.CuArray(Lr)

I = CuArray(collect(1:k+1))
J = CuArray(pr)
V = CUDA.ones(T, k)

println("----------------- Timing CSR /CSC constructor ------------------")
println("CSR directely with CuSparse:")

@time begin
    ipr2_d = CUSPARSE.CuSparseMatrixCSR(I, J, V, (k, prmax))
end

#=
println("CSC directely with CuSparse:")
@time begin
    ipr3_d = CUSPARSE.CuSparseMatrixCSC(I, J, V, (k, prmax))
end
=#

println("Using sparse constructor:")
@time begin
    ipr = sparse(1:k, pr, one(T))
    ipr_d = CUSPARSE.CuSparseMatrixCSR(ipr)
end

println("----------------- Timing sparse x dense --------------------")
csrRowPtr = CuArray(collect(1:k+1))
csrColInd = CuArray(pr)
csrNzVal = CUDA.ones(T, k)

ipr3_d = CUSPARSE.CuSparseMatrixCSR(I, J, V, (k, prmax))

println("CUDA:")
@time begin
    LL = permutedims(Lr_d, (3, 1, 2))
    @cast LL[z, (x, y)] := LL[z, x, y]
    L = transpose(ipr3_d) * LL
    LL = reshape(L, prmax, n, m)
    L1 = permutedims(LL, (2, 3, 1))
end

#=
println("----------------- Timing dense x sparse--------------------")

println("CUDA:")
@time begin
    @cast LL[(x, y), z] := Lr_d[x, y, z]
    L = LL * ipr3_d
    L1 = reshape(L, n, m, prmax)
end
=#

println("CPU:")
@time begin
    @cast L[(x, y), z] := Lr[x, y, z]
    L = L * ipr
    @cast L2[x, y, z] := L[(x, y), z] (y ∈ 1:m)
end

println("Reference (explicit loop):")
@time begin
    L3 = zeros(T, n, m, prmax)
    for i ∈ 1:prmax
        L3[:, :, i] = sum(Lr[:, :, pr.==i], dims=3)
    end
end

@assert Array(L1) ≈ L2 ≈ L3

nothing
