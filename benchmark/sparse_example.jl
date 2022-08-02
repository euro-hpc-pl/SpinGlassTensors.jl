
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using TensorOperations
using TensorCast

T = Float64
n = 100

a = rand(T, n, n); a[a .< T(9//10)] .= T(0)
a_d = CuSparseMatrixCSC(sparse(a))

b = CUDA.rand(T, n, n)

#=
bb = CUDA.rand(T, n, n, n)

@time begin
    @tensor c[x, y, z] := b[x, y, s] * a_d[s, z]
end

@time begin
    @matmul c[x, y, z] := sum(s) bb[x, y, s] * a_d[s, z]
end
=#

@time a_d * b
nothing
