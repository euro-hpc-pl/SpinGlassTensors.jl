using LinearAlgebra
using CUDA


T = Float64
n, m = 10000, 10000

A = rand(T, n, m)
Ad = CuArray(A)

@time q, r = qr(A);
@time qd, rd = qr(Ad);

println(size(q), " ", size(r))
println(size(qd), " ", size(rd))

@assert A ≈ q * r
@assert Ad ≈ qd * rd
