
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using TensorOperations
using TensorCast

CUDA.allowscalar(false)

T = Float64
n = 4096

a = rand(T, n, n); a[a .< T(2//10)] .= T(0)
a_d = CuSparseMatrixCSC(sparse(a))

N = 1000

pr = sort(rand(1:N, N^2))

b = CUDA.rand(T, N^2, n)

#=
bb = CUDA.rand(T, n, n, n)

@time begin
    @tensor c[x, y, z] := b[x, y, s] * a_d[s, z]
end

@time begin
    @matmul c[x, y, z] := sum(s) bb[x, y, s] * a_d[s, z]
end

@time  a_d * b
nothing

@time begin
    a = spzeros(maximum(pr), size(pr, 1))

    for (index, i) in enumerate(pr)
            a[i, index] = 1
        end
        a = CuSparseMatrixCSC(a)

        a*b
    nothing
end
=#
@time begin
    CUDA.memory_status()
    csrRowPtr = CuArray(collect(1:length(pr) + 1))
    CUDA.memory_status()
    csrColInd = CuArray(pr)
    CUDA.memory_status()
    csrNzVal = CUDA.ones(Float64, length(pr))
    CUDA.memory_status()
    a = CUSPARSE.CuSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, (maximum(pr), length(pr)))
    CUDA.memory_status()

    a*b
    CUDA.memory_status()
    nothing
end
