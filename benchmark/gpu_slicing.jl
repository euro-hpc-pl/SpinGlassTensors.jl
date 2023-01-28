using CUDA

T = Float64
n = 10000
k = 500

a = CUDA.rand(T, n, n)
p = reverse(collect(1:k))
p_d = CuArray(p)

@time A = a[:, p]
@time @inbounds A = a[:, p]
@time A = a[:, p_d]
@time @inbounds A = a[:, p_d]
nothing
