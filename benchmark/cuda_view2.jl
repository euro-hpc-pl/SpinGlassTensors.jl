using CUDA, CUDA.CUSPARSE
using NNlib, NNlibCUDA

#println(CUDA.versioninfo())

T = Float64
n = 20
k = 2 ^ 18

p = collect(1:k)
X = CUDA.rand(T, n, n, k)
Y = CUDA.rand(T, n, n, k)

Xp = X[:, :, p]
Yp = Y[:, :, p]

@time out = Y[:, :, p] ⊠ X[:, :, p]

#=
out = CUDA.zeros(T, n, n, k)
@time begin
    for (n, (i, j)) ∈ enumerate(zip(p, p))
        out[:, :, n] = X[:, :, i] * Y[:, :, j]
    end
end
=#
nothing
