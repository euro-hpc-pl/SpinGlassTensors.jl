
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using TensorOperations
using TensorCast

T = Float64
n = 1000

a = rand(T, n, n); a[a .< 0.9] .= T(0)
a = CuSparseMatrixCSC(sparse(a))

b = CUDA.rand(T, n, n)
bb = CUDA.rand(T, n, n, n)

#=
@time begin
    @tensor c[x, y, z] := b[x, y, s] * a[s, z]
end

@time begin
    @matmul c[x, y, z] := sum(s) bb[x, y, s] * a[s, z]
end
=#

@time a * b
nothing
