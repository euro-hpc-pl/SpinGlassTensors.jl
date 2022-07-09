using CUDA
using TensorCast
using TensorOperations

n, m, k = 4, 4, 4
s, r = 4096, 4096

A = rand(n, m, k)
p1 = rand(k, s)
p2 = rand(k, r)

@time @reduce B[x, y, s, r] := sum(z) A[x, y, z] * p1[z, s] * p2[z, r];

@time begin
    @cast a[x, y, _] := p1[x, y]
    @cast b[x, _, y] := p2[x, y]
    p12 = a .* b
    @tensor AA[x, y, u1, u2] := A[x, y, z] * p12[z, u1, u2]
end


A_d = CUDA.CuArray(A)
p1_d = CUDA.CuArray(p1)
p2_d = CUDA.CuArray(p1)

@time begin
    @cast a[x, y, _] := p1_d[x, y]
    @cast b[x, _, y] := p2_d[x, y]
    p12 = a .* b
    @tensor AA[x, y, u1, u2] := A_d[x, y, z] * p12[z, u1, u2]
end

@time @reduce B[x, y, s, r] := sum(z) A_d[x, y, z] * p1_d[z, s] * p2_d[z, r];

nothing
